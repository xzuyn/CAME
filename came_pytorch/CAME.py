import gc
import torch
from torch.optim import Optimizer
import triton
import triton.language as tl


# Reference: https://github.com/NVlabs/Sana/blob/7840b88415bfd1adfff097dc0c3987d45c059660/diffusion/utils/optimizer.py#L537C5-L563C32
@triton.jit
def quantize_kernel_sr(  # stochastic-rounding
    a_ptr,  # fp32
    a_quant_ptr,  # uint8
    scale_ptr,  # fp32
    min_ptr,  # fp32
    seed,  # int
    n_elements,  # int
    NUM_QUANT_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    A_fp32 = tl.load(a_ptr + offsets, mask=mask)

    chunk_min = tl.min(tl.where(mask, A_fp32, float("inf")), axis=0)
    scale = (tl.max(tl.where(mask, A_fp32, float("-inf")), axis=0) - chunk_min) / 255.0
    is_scale_zero = scale == 0.0

    A_fp32 = (A_fp32 - chunk_min) / tl.where(is_scale_zero, 1.0, scale)

    # stochastic rounding to nearest int
    A_fp32 = A_fp32 + tl.rand((seed + offsets + pid).to(tl.uint32), offsets)
    A_fp32 = tl.floor(A_fp32)

    # clamp to 0..255
    A_fp32 = tl.where(is_scale_zero, 0, A_fp32)
    A_fp32 = tl.where(A_fp32 < 0, 0, A_fp32)
    A_fp32 = tl.where(A_fp32 > 255, 255, A_fp32)

    # store quantized bytes (partial store supported by mask)
    tl.store(a_quant_ptr + offsets, A_fp32.to(tl.uint8), mask=mask)

    # store per-block scale & min (only if block exists)
    if pid < NUM_QUANT_BLOCKS:
        tl.store(scale_ptr + pid, scale)
        tl.store(min_ptr + pid, chunk_min)

# Reference: https://github.com/NVlabs/Sana/blob/7840b88415bfd1adfff097dc0c3987d45c059660/diffusion/utils/optimizer.py#L565C5-L582C33
@triton.jit
def dequantize_kernel(
    a_ptr,  # uint8
    a_dequant_ptr,  # fp32
    scale_ptr,  # fp32
    min_ptr,  # fp32
    n_elements,  # int
    QUANT_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    quant_block_idx = offsets // QUANT_BLOCK_SIZE

    A_u8 = tl.load(a_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + quant_block_idx, mask=mask)
    min_val = tl.load(min_ptr + quant_block_idx, mask=mask)

    A_fp32 = A_u8.to(tl.float32)
    A_fp32 = (A_fp32 * scale) + min_val

    tl.store(a_dequant_ptr + offsets, A_fp32, mask=mask)

# Reference: https://github.com/Nerogar/OneTrainer/blob/062443014f380637a2bf8ddaeb2ff9259599ecab/modules/util/bf16_stochastic_rounding.py#L12C1-L57C36
@triton.jit
def add_stochastic_kernel(
    a_ptr,  # bf16
    b_ptr,  # fp32
    alpha,  # float
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # load A and B
    A_bf16 = tl.load(a_ptr + offsets, mask=mask)
    B_fp32 = tl.load(b_ptr + offsets, mask=mask)

    # cast A from bf16 to fp32 (adds 16 empty bits to the mantissa)
    A_fp32 = A_bf16.cast(tl.float32)
    # A + (alpha * B)
    A_fp32 = A_fp32 + (alpha * B_fp32)

    # stochastic rounding to nearest bf16 decimal
    # bitcast A from fp32 to u32 so we can do bit manipulation
    A_u32 = A_fp32.cast(tl.uint32, bitcast=True)
    # create u32 random noise, mask off its upper 16 bits, and add into A
    A_u32 = A_u32 + (tl.randint((seed + offsets + pid).to(tl.uint32), offsets) & 0xFFFF)
    # mask off the lower 16 bits of A
    A_u32 = A_u32 & 0xFFFF0000
    # bitcast the masked A from u32 to fp32
    A_fp32 = A_u32.cast(tl.float32, bitcast=True)
    # cast A from fp32 to bf16 (drop the extra 16 bits in the mantissa)
    A_bf16 = A_fp32.cast(tl.bfloat16)

    tl.store(a_ptr + offsets, A_bf16, mask=mask)


class CAME(Optimizer):
    """
    Implements CAME algorithm with additions.

    This implementation is based on:
      - CAME: Confidence-guided Adaptive Memory Efficient Optimization (https://arxiv.org/abs/2307.02047)
      - Revisiting BFloat16 Training (https://arxiv.org/abs/2010.06192) - Translated to Triton
      - Cautious Optimizers: Improving Training with One Line of Code (https://arxiv.org/abs/2411.16085)
      - SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer
        (https://arxiv.org/abs/2501.18427) - Translated to Triton

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
            update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        enable_stochastic_rounding (bool, optional): utilize stochastic rounding with bfloat16 (default: False)
        enable_cautious (bool, optional): mask out update components whose sign
            conflicts with the gradient, boosting only aligned directions (default: False)
        enable_8bit (bool, optional): enable 8-bit quantization for large layers (default: False)
        block_size (int, optional): quantization block size for 8-bit (default: 2048)
        min_quant_size (int, optional): minimum number of parameters to use quantization (default: 16384)
        enable_gc (bool, optional): enable garbage collection before each step (default: False)
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
        enable_8bit=False,
        block_size=256,
        min_quant_size=16384,
        enable_gc=False,
    ):
        self.torch_gc()

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
            enable_8bit=enable_8bit,
            block_size=block_size,
            min_quant_size=min_quant_size,
            enable_gc=enable_gc,
        )
        super(CAME, self).__init__(params, defaults)

        if not all(
            user_option is False
            for user_option in [enable_stochastic_rounding, enable_cautious, enable_8bit, enable_gc]
        ):
            print("\n==== CAME Modifications ====")
            if enable_stochastic_rounding:
                print(f"- Stochastic Rounding enabled.")
            if enable_cautious:
                print("- Cautious Masking enabled.")
            if enable_8bit:
                print(f"- 8-bit enabled: block_size={block_size}, min_quant_size={min_quant_size}.")
            if enable_gc:
                print("- Garbage Collection enabled.")
            print("==== CAME Modifications ====\n")

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    # https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L523C1-L535C55
    def _should_use_quantization(self, param_shape):
        """Determine if a parameter should be quantized

        Rules:
        1. linear layers: parameter size > min_quant_size
        2. 1x1 conv layers: parameter size > min_quant_size
        3. other layers: use 32bit
        """
        if len(param_shape) == 2:  # linear layer
            return param_shape[0] * param_shape[1] > self.defaults["min_quant_size"]
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:
            return param_shape[0] * param_shape[1] > self.defaults["min_quant_size"]
        return False  # other layers are not quantized

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    # TODO: Implement in triton?
    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def quantize_state_triton_sr(self, A_fp32, block_size):
        n_elements = A_fp32.numel()
        if n_elements <= 1:
            return A_fp32, {}

        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        shape = A_fp32.shape
        num_blocks = (n_elements + block_size - 1) // block_size
        mins = torch.empty((num_blocks,), dtype=torch.float32, device=A_fp32.device)
        scales = torch.empty((num_blocks,), dtype=torch.float32, device=A_fp32.device)
        A_u8 = torch.empty_like(A_fp32, dtype=torch.uint8, device=A_fp32.device)

        assert A_fp32.dtype == torch.float32
        assert A_fp32.is_contiguous()
        assert A_u8.is_contiguous()

        A_fp32 = A_fp32.view(-1)
        A_u8 = A_u8.view(-1)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        quantize_kernel_sr[grid](
            A_fp32,
            A_u8,
            scales,
            mins,
            seed,
            n_elements,
            NUM_QUANT_BLOCKS=num_blocks,
            BLOCK_SIZE=block_size,
        )

        return A_u8, {
            "scales": scales,
            "mins": mins,
            "block_size": block_size,
            "shape": shape,
        }

    def dequantize_state_triton(self, A_u8, quant_state):
        n_elements = A_u8.numel()
        if n_elements <= 1:
            return A_u8

        A_fp32 = torch.empty_like(A_u8, dtype=torch.float32, device=A_u8.device)

        assert A_u8.dtype == torch.uint8
        assert A_u8.is_contiguous()
        assert A_fp32.is_contiguous()

        A_u8 = A_u8.view(-1)
        A_fp32 = A_fp32.view(-1)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        dequantize_kernel[grid](
            A_u8,
            A_fp32,
            quant_state["scales"],
            quant_state["mins"],
            n_elements,
            QUANT_BLOCK_SIZE=quant_state["block_size"],
            BLOCK_SIZE=1024,  # TODO: Tune
        )

        return A_fp32.view(quant_state["shape"])

    def add_stochastic_triton(self, A_bf16, B_fp32, alpha=1.0):
        n_elements = A_bf16.numel()
        if n_elements == 0:
            return A_bf16

        assert A_bf16.shape == B_fp32.shape
        assert A_bf16.dtype == torch.bfloat16
        assert A_bf16.is_contiguous()

        with torch.no_grad():
            if B_fp32.dtype != torch.float32:
                B_fp32 = B_fp32.to(dtype=torch.float32)

            shape = A_bf16.shape

            A_bf16 = A_bf16.view(-1)
            B_fp32 = B_fp32.view(-1)

            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            add_stochastic_kernel[grid](
                A_bf16,
                B_fp32,
                float(alpha),
                seed,
                n_elements,
                BLOCK_SIZE=1024,  # TODO: tune
            )

            return A_bf16.view(shape)

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/torch_util.py
    @staticmethod
    def torch_gc():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @torch.inference_mode()
    def step_param(self, p, group):
        if p.grad is None:
            return

        grad = p.grad.data
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()
        if grad.is_sparse:
            raise RuntimeError("CAME does not support sparse gradients.")

        state = self.state[p]
        grad_shape = grad.shape

        factored = len(grad_shape) >= 2
        use_quantization = group["enable_8bit"] and self._should_use_quantization(grad_shape)

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            # initialize first moment with optional quantization
            if use_quantization:
                state["exp_avg"], state["exp_avg_quant_state"] = self.quantize_state_triton_sr(
                    A_fp32=torch.zeros_like(grad),
                    block_size=group["block_size"],
                )
            else:
                state["exp_avg"] = torch.zeros_like(grad)

            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
            else:
                if use_quantization:
                    state["exp_avg_sq"], state["exp_avg_quant_state"] = self.quantize_state_triton_sr(
                        A_fp32=torch.zeros_like(grad),
                        block_size=group["block_size"],
                    )
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            state["RMS"] = 0

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        # load / dequantize first moment
        if use_quantization:
            exp_avg = self.dequantize_state_triton(
                A_u8=state["exp_avg"],
                quant_state=state["exp_avg_quant_state"]
            )
        else:
            exp_avg = state["exp_avg"]

        update = (grad ** 2) + group["eps"][0]
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
            if use_quantization:
                exp_avg_sq = self.dequantize_state_triton(
                    A_u8=state["exp_avg_sq"],
                    quant_state=state["exp_avg_sq_quant_state"]
                )
            else:
                exp_avg_sq = state["exp_avg_sq"]

            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])

            if use_quantization:
                state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = self.quantize_state_triton_sr(
                    A_fp32=exp_avg_sq,
                    block_size=group["block_size"],
                )
            else:
                state["exp_avg_sq"] = exp_avg_sq

            update = exp_avg_sq.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # update first moment
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

        # re-quantize first moment if using quantization
        if use_quantization:
            state["exp_avg"], state["exp_avg_quant_state"] = self.quantize_state_triton_sr(
                A_fp32=exp_avg,
                block_size=group["block_size"],
            )
        else:
            state["exp_avg"] = exp_avg

        # Confidence-guided strategy
        # Calculation of instability
        res = (update - exp_avg) ** 2 + group["eps"][1]

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

        if group["enable_cautious"]:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)

        if group["weight_decay"] != 0:
            if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
                self.add_stochastic_triton(
                    A_bf16=p.data,
                    B_fp32=p.data.float(),
                    alpha=-group["weight_decay"] * group["lr"],
                )
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
            self.add_stochastic_triton(
                A_bf16=p.data,
                B_fp32=-update.float(),
            )
        else:
            p.data.add_(-update)

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
            if group["enable_gc"]:
                self.torch_gc()

            for p in group["params"]:
                self.step_param(p, group)

        return loss
