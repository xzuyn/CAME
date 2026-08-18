import math

import torch
import torch.optim


class CAME(torch.optim.Optimizer):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
        update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        cautious_weight_decay (boolean, optional):
            Apply Cautious Weight Decay (Chen et al., 2025): only decay coordinates where the
            (pre-learning-rate) optimizer update and the current parameter share the same sign,
            i.e. where decay would not fight the optimizer's own update direction. Only has an
            effect when weight_decay > 0. (default: False)
        cautious_update (boolean, optional):
            Apply Cautious Optimizer modification (Liang et al., 2024): only update coordinates
            where the proposed update direction and the current gradient share the same sign
            (u_t * g_t > 0), rescaled by the active coordinate ratio to prevent magnitude loss.
            (default: False)
        quantize_state (bool, optional):
            (default: False)
        quant_block_size (int, optional):
            (default: 256)
        quant_nbits (int, optional):
            (default: 8)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        cautious_weight_decay=False,
        cautious_update=False,
        quantize_state=False,
        quant_block_size=256,
        quant_nbits=8,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            cautious_weight_decay=cautious_weight_decay,
            cautious_update=cautious_update,
            quantize_state=quantize_state,
            quant_block_size=quant_block_size,
            quant_nbits=quant_nbits,
        )
        super(CAME, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False


    def _get_options(self, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    # https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L523C1-L535C55
    def _should_use_quantization(self, param_shape):
        """Determine if a parameter should be quantized

        Rules:
        1. linear layers: parameter size > min_quant_size
        2. 1x1 conv layers: parameter size > min_quant_size
        3. other layers: use 32bit
        """
        if len(param_shape) == 2:  # linear layer
            return param_shape[0] * param_shape[1] > 16384
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:
            return param_shape[0] * param_shape[1] > 16384
        return False  # other layers are not quantized

    # Reference: https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L537C1-L563C32
    def _quantize_state(self, A, block_size, nbits=8):
        # TODO: better sub-4-bit quantization
        assert 3 <= nbits <= 8, f"nbits must be between 4 and 8 for uint8 storage, got {nbits}"

        n_elements = A.numel()
        if n_elements <= 1:
            return A, {}

        num_blocks = (n_elements + block_size - 1) // block_size

        shape = A.shape
        A = A.flatten().unsqueeze(0)
        pad_len = num_blocks * block_size - n_elements
        if pad_len > 0:
            A = torch.nn.functional.pad(A, (0, pad_len), "replicate")
        A = A.view(num_blocks, block_size)

        qmax = 1.0 if nbits == 1 else float((1 << (nbits - 1)) - 1)
        scales = A.abs().max(dim=1).values / qmax
        is_scale_zero = scales == 0
        safe_scales = torch.where(is_scale_zero, 1.0, scales).unsqueeze(1)

        A.div_(safe_scales)
        A.add_(torch.rand_like(A))
        A.floor_()
        A.clamp_(-qmax, qmax)
        A.masked_fill_(is_scale_zero.unsqueeze(1), 0.0)
        A.add_(qmax)
        A = A.to(torch.uint8)
        A = A.flatten()
        A = A[:n_elements]

        if nbits == 4:
            pack_pad = (-n_elements) % 2
            if pack_pad > 0:
                A = torch.nn.functional.pad(A.unsqueeze(0), (0, pack_pad), "constant", 0).squeeze(0)
            chunks = A.view(-1, 2)
            A = (chunks[:, 0] << 4) | (chunks[:, 1] & 0x0F)
        elif nbits < 8:
            pack_cfg = {
                3: (5, torch.int16, 3, 0x07, 12),  # 5 x 3-bit per int16 (15/16)
                5: (6, torch.int32, 5, 0x1F, 25),  # 6 x 5-bit per int32 (30/32)
                6: (5, torch.int32, 6, 0x3F, 24),  # 5 x 6-bit per int32 (30/32)
                7: (9, torch.int64, 7, 0x7F, 56),  # 9 x 7-bit per int64 (63/64)
            }
            chunk_size, dtype, step, mask, start_shift = pack_cfg[nbits]
            pack_pad = (-n_elements) % chunk_size
            if pack_pad > 0:
                A = torch.nn.functional.pad(A.unsqueeze(0), (0, pack_pad), "constant", 0).squeeze(0)
            chunks = A.to(dtype).view(-1, chunk_size)

            shifts = torch.arange(start_shift, start_shift - chunk_size * step, -step, device=A.device, dtype=dtype)
            A = (((chunks & mask) << shifts).sum(dim=-1)).to(dtype)

        return A, {
            "scales": scales,
            "qmax": qmax,
            "block_size": block_size,
            "shape": shape,
            "n_elements": n_elements,
            "nbits": nbits,
        }

    # Reference: https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L565C1-L582C33
    def _dequantize_state(self, A, quant_state):
        n_elements = quant_state.get("n_elements", A.numel())
        if n_elements <= 1:
            return A

        nbits = quant_state.get("nbits", 8)
        if nbits == 4:
            unpacked = torch.empty((A.numel() * 2,), dtype=torch.uint8, device=A.device)
            unpacked[0::2] = (A >> 4) & 0x0F
            unpacked[1::2] = A & 0x0F
            A = unpacked[:n_elements]
        elif nbits < 8:
            pack_cfg = {
                3: (5, torch.int16, 3, 0x07, 12),  # 5 x 3-bit per int16 (15/16)
                5: (6, torch.int32, 5, 0x1F, 25),  # 6 x 5-bit per int32 (30/32)
                6: (5, torch.int32, 6, 0x3F, 24),  # 5 x 6-bit per int32 (30/32)
                7: (9, torch.int64, 7, 0x7F, 56),  # 9 x 7-bit per int64 (63/64)
            }
            chunk_size, dtype, step, mask, start_shift = pack_cfg[nbits]
            shifts = torch.arange(start_shift, start_shift - chunk_size * step, -step, device=A.device, dtype=dtype)
            A = ((A.unsqueeze(-1) >> shifts) & mask).to(torch.uint8)
            A = A.flatten()[:n_elements]

        block_size = quant_state["block_size"]
        num_blocks = (n_elements + block_size - 1) // block_size

        pad_len = num_blocks * block_size - n_elements
        if pad_len > 0:
            A = torch.nn.functional.pad(A.unsqueeze(0), (0, pad_len), "constant", 0).squeeze(0)

        A = A.view(num_blocks, block_size).float()
        A.sub_(quant_state["qmax"])
        A.mul_(quant_state["scales"].unsqueeze(1))
        A = A.flatten()[:n_elements]

        return A.reshape(quant_state["shape"])

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = self._get_options(grad_shape)
                use_quantization = group["quantize_state"] and self._should_use_quantization(grad_shape)

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_quantization:
                        state["exp_avg"], state["exp_avg_quant_state"] = self._quantize_state(
                            A=torch.zeros_like(grad),
                            block_size=group["quant_block_size"],
                            nbits=group["quant_nbits"],
                        )
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)

                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).type_as(grad)
                    else:
                        if use_quantization:
                            state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = self._quantize_state(
                                A=torch.zeros_like(grad),
                                block_size=group["quant_block_size"],
                                nbits=group["quant_nbits"],
                            )
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(group["betas"][1]).add_(
                        update.mean(dim=-1), alpha=1.0 - group["betas"][1]
                    )
                    exp_avg_sq_col.mul_(group["betas"][1]).add_(
                        update.mean(dim=-2), alpha=1.0 - group["betas"][1]
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    if use_quantization:
                        exp_avg_sq = self._dequantize_state(
                            A=state["exp_avg_sq"],
                            quant_state=state["exp_avg_sq_quant_state"]
                        )
                    else:
                        exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])

                    if use_quantization:
                        state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = self._quantize_state(
                            A=exp_avg_sq,
                            block_size=group["quant_block_size"],
                            nbits=group["quant_nbits"],
                        )
                    else:
                        state["exp_avg_sq"] = exp_avg_sq

                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )

                if use_quantization:
                    exp_avg = self._dequantize_state(
                        A=state["exp_avg"],
                        quant_state=state["exp_avg_quant_state"]
                    )
                else:
                    exp_avg = state["exp_avg"]

                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg)**2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    exp_avg_res_row.mul_(group["betas"][2]).add_(
                        res.mean(dim=-1), alpha=1.0 - group["betas"][2]
                    )
                    exp_avg_res_col.mul_(group["betas"][2]).add_(
                        res.mean(dim=-2), alpha=1.0 - group["betas"][2]
                    )

                    # Approximation of exponential moving average of instability
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if use_quantization:
                    state["exp_avg"], state["exp_avg_quant_state"] = self._quantize_state(
                        A=exp_avg,
                        block_size=group["quant_block_size"],
                        nbits=group["quant_nbits"],
                    )
                else:
                    state["exp_avg"] = exp_avg

                if group["cautious_update"]:
                    # Cautious Optimizer (Liang et al., 2024): zero out coordinates where
                    # update conflicts with gradient (u_t * g_t <= 0) and scale by active ratio.
                    mask = (update * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    update.mul_(mask)

                if group["weight_decay"] != 0:
                    if group["cautious_weight_decay"]:
                        # Cautious Weight Decay (Chen et al., 2025): only decay where the
                        # update and the parameter share a sign, i.e. u_t * x_t >= 0.
                        mask = (update * p.data >= 0).to(p.dtype)
                        p.data.add_(
                            p.data * mask, alpha=-group["weight_decay"] * group["lr"]
                        )
                    else:
                        p.data.add_(
                            p.data, alpha=-group["weight_decay"] * group["lr"]
                        )

                update.mul_(group["lr"])
                p.data.add_(-update)

        return loss
