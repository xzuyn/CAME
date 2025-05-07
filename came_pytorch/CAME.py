import math

import numpy as np
import torch
from torch.optim import Optimizer


class CAME(Optimizer):
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
        enable_stochastic_rounding (bool, optional): utilize stochastic rounding with bfloat16 (default: False)
        enable_cautious (bool, optional): mask out update components whose sign
            conflicts with the gradient, boosting only aligned directions (default: False)
        enable_grams (bool, optional): replace each updates direction with the
            gradients sign while preserving its adaptive magnitude (default: False)
        enable_8bit (bool, optional): enable 8-bit quantization for large layers (default: False)
        block_size (int, optional): quantization block size for 8-bit (default: 2048)
        min_8bit_size (int, optional): minimum number of parameters to use 8-bit (default: 16384)
        quiet_8bit (bool, optional): don't print layer info (default: True)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        enable_stochastic_rounding=False,
        enable_cautious=False,
        enable_grams=False,
        enable_8bit=False,
        block_size=2048,
        min_8bit_size=16384,
        quiet_8bit=True,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            enable_stochastic_rounding=enable_stochastic_rounding,
            enable_cautious=enable_cautious,
            enable_grams=enable_grams,
            enable_8bit=enable_8bit,
            block_size=block_size,
            min_8bit_size=min_8bit_size,
            quiet_8bit=quiet_8bit,
        )
        super(CAME, self).__init__(params, defaults)

        if enable_stochastic_rounding:
            print("Using Stochastic Rounding for CAME.")
        if enable_cautious:
            print("Using Cautious Masking for CAME.")
        if enable_grams:
            print("Using Grams for CAME.")
        if enable_8bit:
            print(f"Initializing CAME with 8-bit support: block_size={block_size}, min_8bit_size={min_8bit_size}")

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    @staticmethod
    def _get_options(param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _should_use_8bit(self, param_shape):
        """Determines whether parameters should be quantized to 8-bit based on size and layer type"""
        if len(param_shape) == 2:  # Linear layers
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:  # 1x1 conv
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        return False

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
    @staticmethod
    def _copy_stochastic(tensor_target, tensor_source):
        """
        Copies tensor_source into tensor_target using stochastic rounding

        Args:
            tensor_target: the target tensor with dtype=bfloat16
            tensor_source: the source tensor with dtype=float32
        """
        # Reinterpret float32 bits as int32
        source_bits = tensor_source.view(torch.int32)

        # Generate uniform random noise for the lower 16 mantissa bits
        noise = torch.randint(
            low=0,
            high=1 << 16,
            size=source_bits.shape,
            dtype=torch.int32,
            device=source_bits.device,
        )

        # Add noise to the LSBs of the mantissa
        noisy = source_bits + noise

        # Mask off the lower 16 bits
        noisy.bitwise_and_(-65536)

        # Reinterpret bits back to float32, then cast to target dtype
        rounded = noisy.view(torch.float32).to(dtype=tensor_target.dtype)
        tensor_target.copy_(rounded)

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
    def _add_stochastic(self, tensor_input, tensor_other, alpha=1.0):
        """
        Adds tensor_other to tensor_input using stochastic rounding

        Args:
            tensor_input: the input tensor with dtype=bfloat16
            tensor_other: the other tensor to add (dtype float32 or bfloat16)
            alpha: scaling factor for tensor_other
        """
        # Compute the sum in float32
        summed = tensor_input.to(torch.float32) + (tensor_other.to(torch.float32) * alpha)

        # Copy sum into tensor_input using stochastic rounding
        self._copy_stochastic(tensor_input, summed)

    def _quantize_state(self, state_tensor, block_size):
        """Quantizes the state tensor to 8-bit with simple min-max per block"""
        if state_tensor.numel() <= 1:
            return state_tensor
        quantized_chunks = []
        for chunk in state_tensor.split(block_size):
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            scale = (chunk_max - chunk_min) / 255
            quantized_data = ((chunk - chunk_min) / scale).round().byte()
            quantized_chunks.append({"data": quantized_data, "scale": scale, "min": chunk_min})
        return quantized_chunks

    def _dequantize_state(self, quantized_chunks):
        """Dequantizes quantized chunks back to float32"""
        if not isinstance(quantized_chunks, list):
            return quantized_chunks
        chunks = []
        for c in quantized_chunks:
            chunks.append(c["data"].float() * c["scale"] + c["min"])
        return torch.cat(chunks)

    def print_layer_info(self, param_shape, use_8bit):
        size = np.prod(param_shape)
        layer_type = "unknown"
        if len(param_shape) == 1:
            layer_type = "1D Layer"
        elif len(param_shape) == 2:
            layer_type = "Linear"
        elif len(param_shape) == 4:
            if param_shape[2] == 1 and param_shape[3] == 1:
                layer_type = "1x1 Conv"
            else:
                layer_type = "Conv"
        status = "8bit" if use_8bit else "32bit"
        print(f"{layer_type} layer with shape {param_shape}: {size:,} params -> using {status}")

    @torch.inference_mode()
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
                use_8bit = group["enable_8bit"] and self._should_use_8bit(grad_shape)

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    # initialize first moment with optional 8-bit quantization
                    if not group["quiet_8bit"]:
                        self.print_layer_info(grad_shape, use_8bit)
                    if use_8bit:
                        state["exp_avg"] = self._quantize_state(torch.zeros_like(grad), group["block_size"])
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                    else:
                        if use_8bit:
                            state["exp_avg_sq"] = self._quantize_state(torch.zeros_like(grad), group["block_size"])
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                # load / dequantize first moment
                if use_8bit:
                    exp_avg = self._dequantize_state(state["exp_avg"])
                else:
                    exp_avg = state["exp_avg"]

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(group["betas"][1]).add_(update.mean(dim=-1), alpha=1.0 - group["betas"][1])
                    exp_avg_sq_col.mul_(group["betas"][1]).add_(update.mean(dim=-2), alpha=1.0 - group["betas"][1])

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    # non-factored: update second moment, quantize if needed
                    if use_8bit:
                        exp_avg_sq = self._dequantize_state(state["exp_avg_sq"])
                    else:
                        exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    if use_8bit:
                        state["exp_avg_sq"] = self._quantize_state(exp_avg_sq, group["block_size"])
                    else:
                        state["exp_avg_sq"] = exp_avg_sq
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # update first moment
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])
                # re-quantize first moment if using 8bit
                if use_8bit:
                    state["exp_avg"] = self._quantize_state(exp_avg, group["block_size"])
                else:
                    state["exp_avg"] = exp_avg

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg)**2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    exp_avg_res_row.mul_(group["betas"][2]).add_(res.mean(dim=-1), alpha=1.0 - group["betas"][2])
                    exp_avg_res_col.mul_(group["betas"][2]).add_(res.mean(dim=-2), alpha=1.0 - group["betas"][2])

                    # Approximation of exponential moving average of instability
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                # Cautious masking - https://arxiv.org/abs/2411.16085
                if group["enable_cautious"]:
                    mask = (update * grad > 0).to(grad.dtype)
                    mask.div_(mask.mean().clamp_(min=1e-3))
                    update = update * mask

                # Grams: adaptive momentum scaling - https://arxiv.org/abs/2412.17107
                if group["enable_grams"]:
                    update = update.abs_().mul_(grad.sign_())

                if group["weight_decay"] != 0:
                    if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
                        self._add_stochastic(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
                    else:
                        p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                update.mul_(group["lr"])
                if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
                    self._add_stochastic(p.data, -update)
                else:
                    p.data.add_(-update)

        return loss
