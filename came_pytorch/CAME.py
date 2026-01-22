import gc
import torch
from torch.optim import Optimizer
import triton
import triton.language as tl
import random


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

    # Load A and cast from bf16 to fp32 (adds 16 empty bits to the mantissa)
    A_fp32 = tl.load(a_ptr + offsets, mask=mask, eviction_policy="evict_first").cast(tl.float32)

    # Load B
    B_fp32 = tl.load(b_ptr + offsets, mask=mask, eviction_policy="evict_first")

    # A + (alpha * B)
    A_fp32 = A_fp32 + (alpha * B_fp32)

    # stochastic rounding to nearest bf16 decimal
    # bitcast A from fp32 to u32 so we can do bit manipulation
    A_u32 = A_fp32.cast(tl.uint32, bitcast=True)
    # create u32 random noise, mask off its upper 16 bits, and add into A
    seeds = seed + offsets + (pid * BLOCK_SIZE)
    A_u32 = A_u32 + (tl.randint(seeds.to(tl.uint32), offsets) & 0xFFFF)
    # mask off the lower 16 bits of A
    A_u32 = A_u32 & 0xFFFF0000
    # bitcast the masked A from u32 to fp32
    A_fp32 = A_u32.cast(tl.float32, bitcast=True)
    # cast A from fp32 to bf16 (drop the extra 16 bits in the mantissa)
    A_bf16 = A_fp32.cast(tl.bfloat16)

    tl.store(a_ptr + offsets, A_bf16, mask=mask)


@triton.jit
def fused_update_exp_avg_sq_kernel(
    grad_ptr,  # fp32 or bf16
    update_sq_ptr,  # fp32
    exp_avg_sq_ptr,  # uint8
    scale_ptr,  # fp32
    min_ptr,  # fp32
    beta,  # float
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes exactly one quantization block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load metadata (scalars per block)
    old_scale = tl.load(scale_ptr + pid)
    old_min = tl.load(min_ptr + pid)

    # Load quantized state and convert to fp32
    state_fp32 = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    # Dequantize
    state_fp32 = (state_fp32 * old_scale) + old_min

    # Load update
    update_sq_val = tl.load(update_sq_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")

    # Update EMA: exp_avg_sq.mul_(beta).add_(update, alpha=1-beta)
    state_fp32 = (state_fp32 * beta) + (update_sq_val * (1.0 - beta))

    # Load grad
    grad_val = tl.load(grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")

    # update = exp_avg_sq.rsqrt().mul_(grad)
    output_val = tl.rsqrt(state_fp32 + 1e-10) * grad_val

    # Store update
    tl.store(update_sq_ptr + offsets, output_val, mask=mask)

    # Calculate new min/max for this quantization block
    chunk_min = tl.min(tl.where(mask, state_fp32, float("inf")), axis=0)
    chunk_max = tl.max(tl.where(mask, state_fp32, float("-inf")), axis=0)

    scale = tl.maximum((chunk_max - chunk_min) / 255.0, 1e-10)

    # Normalize
    state_norm = (state_fp32 - chunk_min) / scale

    # Stochastic Rounding
    seeds = seed + offsets + (pid * BLOCK_SIZE)
    state_norm = state_norm + tl.rand(seeds.to(tl.uint32), offsets)
    state_norm = tl.floor(state_norm)

    # Clamp to 0..255
    state_norm = tl.clamp(state_norm, 0.0, 255.0)

    # Store State (uint8)
    tl.store(exp_avg_sq_ptr + offsets, state_norm.to(tl.uint8), mask=mask)

    # Store Metadata
    tl.store(scale_ptr + pid, scale)
    tl.store(min_ptr + pid, chunk_min)


@triton.jit
def fused_update_exp_avg_kernel(
    update_ptr,  # fp32
    exp_avg_ptr,  # uint8
    scale_ptr,  # fp32
    min_ptr,  # fp32
    output_ptr,  # fp32
    beta,  # float
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes exactly one quantization block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load metadata (scalars per block)
    old_scale = tl.load(scale_ptr + pid)
    old_min = tl.load(min_ptr + pid)

    # Load quantized state and convert to fp32
    state_fp32 = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    # Dequantize
    state_fp32 = (state_fp32 * old_scale) + old_min

    # Load update
    update_val = tl.load(update_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")

    # Update EMA: exp_avg.mul_(beta).add_(update, alpha=1-beta)
    state_fp32 = (state_fp32 * beta) + (update_val * (1.0 - beta))

    # Store the updated exp_avg (fp32) for subsequent instability calculation
    tl.store(output_ptr + offsets, state_fp32, mask=mask)

    # Calculate new min/max for this quantization block
    chunk_min = tl.min(tl.where(mask, state_fp32, float("inf")), axis=0)
    chunk_max = tl.max(tl.where(mask, state_fp32, float("-inf")), axis=0)

    scale = tl.maximum((chunk_max - chunk_min) / 255.0, 1e-10)

    # Normalize
    state_norm = (state_fp32 - chunk_min) / scale

    # Stochastic Rounding
    seeds = seed + offsets + (pid * BLOCK_SIZE)
    state_norm = state_norm + tl.rand(seeds.to(tl.uint32), offsets)
    state_norm = tl.floor(state_norm)

    # Clamp to 0..255
    state_norm = tl.clamp(state_norm, 0.0, 255.0)

    # Store State (uint8)
    tl.store(exp_avg_ptr + offsets, state_norm.to(tl.uint8), mask=mask)

    # Store Metadata (only first thread in block writes)
    if tl.program_id(axis=0) == pid:
        tl.store(scale_ptr + pid, scale)
        tl.store(min_ptr + pid, chunk_min)


def add_stochastic_triton(A_bf16, B, alpha=1.0):
    n_elements = A_bf16.numel()
    if n_elements == 0:
        return A_bf16

    assert A_bf16.shape == B.shape
    assert A_bf16.dtype == torch.bfloat16
    assert A_bf16.is_contiguous()

    with torch.no_grad():
        B = B.contiguous()

        shape = A_bf16.shape

        A_bf16 = A_bf16.view(-1)
        B = B.view(-1)

        seed = random.randint(0, 2**32 - 1)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        add_stochastic_kernel[grid](
            A_bf16,
            B,
            float(alpha),
            seed,
            n_elements,
            BLOCK_SIZE=1024,  # TODO: tune
        )

        return A_bf16.view(shape)


def fused_update_exp_avg_sq_triton(
    grad,
    update_sq,
    exp_avg_sq_u8,
    quant_state,
    beta,
):
    """
    Fuses: Dequantize -> EMA Update -> Compute Output -> Quantize

    `update_sq` is modified in-place

    Should be ~2-4x faster, and use ~0.5x peak memory usage
    """

    assert update_sq.dtype == torch.float32, (
        "update_sq must be float32 as it will be stored as float32"
    )
    assert update_sq.is_contiguous(), (
        "update_sq must be contiguous as it is modified in-place"
    )

    n_elements = grad.numel()

    # Flatten views for kernel
    grad_flat = grad.view(-1)
    update_sq_flat = update_sq.view(-1)
    state_u8_flat = exp_avg_sq_u8.view(-1)

    # Quantization metadata
    scales = quant_state["scales"]
    mins = quant_state["mins"]
    block_size = quant_state["block_size"]
    num_blocks = scales.numel()

    seed = random.randint(0, 2**32 - 1)

    # Launch one thread block per quantization block
    grid = (num_blocks,)

    fused_update_exp_avg_sq_kernel[grid](
        grad_flat,
        update_sq_flat,
        state_u8_flat,
        scales,
        mins,
        float(beta),
        seed,
        n_elements,
        BLOCK_SIZE=block_size,
    )


def fused_update_exp_avg_triton(
    update,
    exp_avg_u8,
    quant_state,
    beta,
):
    """
    Fuses: Dequantize -> EMA Update -> Quantize

    Returns: The updated exp_avg (float32) for subsequent instability calculation
    """
    n_elements = update.numel()

    assert update.dtype == torch.float32
    assert update.is_contiguous()

    output = torch.empty_like(update, dtype=torch.float32)

    # Flatten views for kernel
    update_flat = update.view(-1)
    output_flat = output.view(-1)
    state_u8_flat = exp_avg_u8.view(-1)

    # Quantization metadata
    scales = quant_state["scales"]
    mins = quant_state["mins"]
    block_size = quant_state["block_size"]
    num_blocks = scales.numel()

    seed = random.randint(0, 2**32 - 1)

    # Launch one thread block per quantization block
    grid = (num_blocks,)

    fused_update_exp_avg_kernel[grid](
        update_flat,
        state_u8_flat,
        scales,
        mins,
        output_flat,
        float(beta),
        seed,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return output


# TODO: Implement in triton?
def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True)
    r_factor = (exp_avg_sq_row / row_mean).rsqrt().unsqueeze(-1)
    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
    return torch.mul(r_factor, c_factor)


class CAME(Optimizer):
    """
    Implements CAME algorithm with additions.

    This implementation is based on:
      - CAME: Confidence-guided Adaptive Memory Efficient Optimization (https://arxiv.org/abs/2307.02047)
      - Revisiting BFloat16 Training (https://arxiv.org/abs/2010.06192) - Translated to Triton
      - Cautious Optimizers: Improving Training with One Line of Code (https://arxiv.org/abs/2411.16085)
      - Cautious Weight Decay (https://arxiv.org/abs/2510.12402)
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
        enable_cautious_update (bool, optional): mask out update components whose sign
            conflicts with the gradient, boosting only aligned directions (default: False)
        enable_cautious_weight_decay (bool, optional): only apply weight decay when the parameter
            and the update have the same sign (default: False)
        enable_8bit (bool, optional): enable 8-bit quantization for large layers (default: False)
        block_size (int, optional): quantization block size for 8-bit (default: 2048)
        min_quant_size (int, optional): minimum number of parameters to use quantization (default: 16384)
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
        enable_cautious_update=False,
        enable_cautious_weight_decay=False,
        enable_8bit=False,
        block_size=256,
        min_quant_size=16384,
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
            enable_cautious_update=enable_cautious_update,
            enable_cautious_weight_decay=enable_cautious_weight_decay,
            enable_8bit=enable_8bit,
            block_size=block_size,
            min_quant_size=min_quant_size,
        )
        super(CAME, self).__init__(params, defaults)

        if not all(
            user_option is False
            for user_option in [
                enable_stochastic_rounding,
                enable_cautious_update,
                enable_cautious_weight_decay,
                enable_8bit,
            ]
        ):
            print("\n==== CAME Modifications ====")
            if enable_stochastic_rounding:
                print(f"- Stochastic Rounding enabled.")
            if enable_cautious_update:
                print("- Cautious Update enabled.")
            if enable_cautious_weight_decay:
                print("- Cautious Weight Decay enabled.")
            if enable_8bit:
                print(
                    f"- 8-bit enabled: block_size={block_size}, min_quant_size={min_quant_size}."
                )
            print("==== CAME Modifications ====\n")

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

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
        use_quantization = group["enable_8bit"] and grad.numel() > group["min_quant_size"]

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            # initialize first moment with optional quantization
            if use_quantization:
                n_elements = grad.numel()
                block_size = group["block_size"]
                num_blocks = (n_elements + block_size - 1) // block_size

                state["exp_avg"] = torch.zeros_like(grad, dtype=torch.uint8)
                state["exp_avg_quant_state"] = {
                    "scales": torch.zeros(
                        num_blocks, dtype=torch.float32, device=grad.device
                    ),
                    "mins": torch.zeros(
                        num_blocks, dtype=torch.float32, device=grad.device
                    ),
                    "block_size": block_size,
                    "shape": grad_shape,
                }
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
                    n_elements = grad.numel()
                    block_size = group["block_size"]
                    num_blocks = (n_elements + block_size - 1) // block_size

                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.uint8)
                    state["exp_avg_sq_quant_state"] = {
                        "scales": torch.zeros(
                            num_blocks, dtype=torch.float32, device=grad.device
                        ),
                        "mins": torch.zeros(
                            num_blocks, dtype=torch.float32, device=grad.device
                        ),
                        "block_size": block_size,
                        "shape": grad_shape,
                    }
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)

        state["step"] += 1

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
            update = _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        else:
            # non-factored: update second moment
            if use_quantization:
                # update is modified in-place
                # update must be fp32 and contiguous
                fused_update_exp_avg_sq_triton(
                    grad=grad,
                    update_sq=update,
                    exp_avg_sq_u8=state["exp_avg_sq"],
                    quant_state=state["exp_avg_sq_quant_state"],
                    beta=group["betas"][1],
                )
            else:
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(group["betas"][1]).add_(
                    update, alpha=1.0 - group["betas"][1]
                )
                update = exp_avg_sq.rsqrt().mul_(grad)
                state["exp_avg_sq"] = exp_avg_sq

        update.div_(
            (
                (update.norm(2) / (update.numel() ** 0.5)) / group["clip_threshold"]
            ).clamp_(min=1.0)
        )

        # update first moment
        if use_quantization:
            exp_avg = fused_update_exp_avg_triton(
                update=update,
                exp_avg_u8=state["exp_avg"],
                quant_state=state["exp_avg_quant_state"],
                beta=group["betas"][0],
            )
        else:
            exp_avg = state["exp_avg"]
            exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

        masked_exp_avg = exp_avg
        if group["enable_cautious_update"]:
            mask = (masked_exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            masked_exp_avg = masked_exp_avg * mask

        if factored:
            # Confidence-guided strategy
            # Calculation of instability
            res = (update - masked_exp_avg) ** 2 + group["eps"][1]

            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]

            exp_avg_res_row.mul_(group["betas"][2]).add_(
                res.mean(dim=-1), alpha=1.0 - group["betas"][2]
            )
            exp_avg_res_col.mul_(group["betas"][2]).add_(
                res.mean(dim=-2), alpha=1.0 - group["betas"][2]
            )

            # Approximation of exponential moving average of instability
            res_approx = _approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
            confidence_factor = res_approx
            update = res_approx.mul_(masked_exp_avg)
        else:
            update = masked_exp_avg.clone()

        if not use_quantization:
            state["exp_avg"] = exp_avg

        if group["weight_decay"] != 0:
            decay_src = p.data
            if group["enable_cautious_weight_decay"]:
                mask = (update * p.data >= 0)
                decay_src = decay_src * mask
                if factored:
                    decay_src.mul_(confidence_factor)

            update.add_(decay_src, alpha=group["weight_decay"])

        if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
            add_stochastic_triton(
                A_bf16=p.data,
                B=update,
                alpha=-group["lr"],
            )
        else:
            p.data.add_(update, alpha=-group["lr"])

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
                self.step_param(p, group)

        return loss
