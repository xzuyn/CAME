import gc
import torch
from torch.optim import Optimizer
import triton
import triton.language as tl
import random
import math


# Reference: https://github.com/Nerogar/OneTrainer/blob/062443014f380637a2bf8ddaeb2ff9259599ecab/modules/util/bf16_stochastic_rounding.py#L12C1-L57C36
@triton.jit
def add_stochastic_kernel(
    a_ptr,  # bf16 or fp32
    b_ptr,  # fp32
    alpha,  # float
    bias_correction,  # float
    weight_decay,  # float
    enable_cautious_weight_decay: tl.constexpr,  # bool
    enable_stochastic_rounding: tl.constexpr,  # bool
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid.to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load A (p.data) and cast from bf16 to fp32 (adds 16 empty bits to the mantissa)
    A_fp32 = tl.load(a_ptr + offsets, mask=mask).cast(tl.float32)

    # Load B (update)
    B_fp32 = tl.load(b_ptr + offsets, mask=mask) / bias_correction

    if enable_cautious_weight_decay:  # TODO: include confidence_factor?
        B_fp32 = B_fp32 + (A_fp32 * (B_fp32 * A_fp32 >= 0)) * weight_decay
    else:
        B_fp32 = B_fp32 + (A_fp32 * weight_decay)

    # A + (alpha * B)
    A_fp32 = A_fp32 + (alpha * B_fp32)

    if enable_stochastic_rounding:
        # stochastic rounding to nearest bf16 decimal
        # bitcast A from fp32 to u32 so we can do bit manipulation
        A_u32 = A_fp32.cast(tl.uint32, bitcast=True)
        # create u32 random noise, mask off its upper 16 bits, and add into A
        A_u32 = A_u32 + tl.randint(seed, offsets) & 0xFFFF
        # mask off the lower 16 bits of A
        A_u32 = A_u32 & 0xFFFF0000
        # bitcast the masked A from u32 to fp32
        A_fp32 = A_u32.cast(tl.float32, bitcast=True)
        # cast A from fp32 to bf16 (drop the extra 16 bits in the mantissa)
        A_bf16 = A_fp32.cast(tl.bfloat16)
        tl.store(a_ptr + offsets, A_bf16, mask=mask)
    else:
        tl.store(a_ptr + offsets, A_fp32, mask=mask)


@triton.jit
def fused_update_exp_avg_sq_kernel(
    grad_ptr,  # fp32 or bf16
    update_sq_ptr,  # fp32
    exp_avg_sq_ptr,  # uint8
    scale_ptr,  # fp32
    min_ptr,  # fp32
    beta,  # float
    eps,  # float
    bias_correction,  # float
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes exactly one quantization block
    pid = tl.program_id(axis=0)
    block_start = pid.to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load metadata (scalars per block)
    old_scale = tl.load(scale_ptr + pid)
    old_min = tl.load(min_ptr + pid)

    # Load quantized state and convert to fp32
    state_fp32 = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Dequantize
    state_fp32 = (state_fp32 * old_scale) + old_min

    # Load grad
    grad_val = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # update = (grad**2) + group["eps"][0]
    update_sq_val = (grad_val * grad_val) + eps

    # Update EMA: exp_avg_sq.mul_(beta).add_(update, alpha=1-beta)
    state_fp32 = (state_fp32 * beta) + (update_sq_val * (1.0 - beta))

    # update = exp_avg_sq.rsqrt().mul_(grad) with bias correction
    output_val = tl.rsqrt(state_fp32 / bias_correction) * grad_val

    # Store update
    tl.store(update_sq_ptr + offsets, output_val, mask=mask)

    # Calculate new min/max for this quantization block
    chunk_min = tl.min(tl.where(mask, state_fp32, 1e30), axis=0)
    chunk_max = tl.max(tl.where(mask, state_fp32, -1e30), axis=0)

    scale = (chunk_max - chunk_min) / 255.0
    is_scale_zero = scale == 0.0

    # Normalize
    state_norm = (state_fp32 - chunk_min) / tl.where(is_scale_zero, 1.0, scale)

    # Stochastic Rounding
    state_norm = state_norm + tl.rand(seed, offsets)
    state_norm = tl.floor(state_norm)

    # Clamp to 0..255
    state_norm = tl.clamp(state_norm, 0, 255)

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
    rms_clip_scale,  # float
    seed,  # int
    n_elements,  # int
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread block processes exactly one quantization block
    pid = tl.program_id(axis=0)
    block_start = pid.to(tl.int64) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load metadata (scalars per block)
    old_scale = tl.load(scale_ptr + pid)
    old_min = tl.load(min_ptr + pid)

    # Load quantized state and convert to fp32
    state_fp32 = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Dequantize
    state_fp32 = (state_fp32 * old_scale) + old_min

    # Load update
    update_val = tl.load(update_ptr + offsets, mask=mask, other=0.0)

    # Apply RMS clipping scale
    update_val = update_val * rms_clip_scale

    # Update EMA: exp_avg.mul_(beta).add_(update, alpha=1-beta)
    state_fp32 = (state_fp32 * beta) + (update_val * (1.0 - beta))

    # Store the updated exp_avg (fp32) for subsequent instability calculation
    tl.store(output_ptr + offsets, state_fp32, mask=mask)

    # Calculate new min/max for this quantization block
    chunk_min = tl.min(tl.where(mask, state_fp32, 1e30), axis=0)
    chunk_max = tl.max(tl.where(mask, state_fp32, -1e30), axis=0)

    scale = (chunk_max - chunk_min) / 255.0
    is_scale_zero = scale == 0.0

    # Normalize
    state_norm = (state_fp32 - chunk_min) / tl.where(is_scale_zero, 1.0, scale)

    # Stochastic Rounding
    state_norm = state_norm + tl.rand(seed, offsets)
    state_norm = tl.floor(state_norm)

    # Clamp to 0..255
    state_norm = tl.clamp(state_norm, 0, 255)

    # Store State (uint8)
    tl.store(exp_avg_ptr + offsets, state_norm.to(tl.uint8), mask=mask)

    # Store Metadata
    tl.store(scale_ptr + pid, scale)
    tl.store(min_ptr + pid, chunk_min)


def apply_update_triton(
    A,
    B,
    alpha=1.0,
    bias_correction=1.0,
    weight_decay=0.0,
    enable_cautious_weight_decay=False,
    enable_stochastic_rounding=False,
):
    n_elements = A.numel()
    if n_elements == 0:
        return A

    with torch.no_grad():
        B = B.contiguous()
        shape = A.shape
        A_flat = A.view(-1)
        B_flat = B.view(-1)

        seed = random.randint(0, 2**32 - 1)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        do_sr = enable_stochastic_rounding and A.dtype == torch.bfloat16

        add_stochastic_kernel[grid](
            A_flat,
            B_flat,
            float(alpha),
            float(bias_correction),
            float(weight_decay),
            bool(enable_cautious_weight_decay),
            bool(do_sr),
            seed,
            n_elements,
            BLOCK_SIZE=1024,  # TODO: tune
            num_warps=8  # TODO: leave at default?
        )

        return A_flat.view(shape)


def fused_update_exp_avg_sq_triton(
    grad,
    exp_avg_sq_u8,
    quant_state,
    beta,
    eps,
    bias_correction,
):
    n_elements = grad.numel()

    update_sq = torch.empty_like(grad, dtype=torch.float32)

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
        float(eps),
        float(bias_correction),
        seed,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return update_sq


def fused_update_exp_avg_triton(
    update,
    exp_avg_u8,
    quant_state,
    beta,
    rms_clip_scale,
):
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
        float(rms_clip_scale),
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
      - Revisiting BFloat16 Training (https://arxiv.org/abs/2010.06192)
      - Cautious Optimizers: Improving Training with One Line of Code (https://arxiv.org/abs/2411.16085)
      - Cautious Weight Decay (https://arxiv.org/abs/2510.12402)
      - SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer (https://arxiv.org/abs/2501.18427)
      - OrthoGrad Improves Neural Calibration (https://www.arxiv.org/abs/2506.04487)
      - Prodigy: An Expeditiously Adaptive Parameter-Free Learner (https://arxiv.org/abs/2306.06101)
      - The Road Less Scheduled (https://arxiv.org/abs/2405.15682)

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
        enable_8bit (bool, optional): enable fused 8-bit quantization for large layers (default: False)
        block_size (int, optional): quantization block size for 8-bit (default: 256)
        min_quant_size (int, optional): minimum number of parameters to use quantization (default: 16384)
        enable_orthograd (bool, optional): project gradients orthogonally to the weights to prevent
            magnitude inflation (default: False)
        enable_prodigy (bool, optional): automatically estimate and scale the learning rate using
            the Prodigy method; effective lr becomes lr * d at each step (default: False)
        d0 (float, optional): initial lower-bound estimate of the distance to solution D;
            only used when enable_prodigy=True (default: 1e-6)
        enable_schedule_free (bool, optional): enable Schedule-Free optimization to remove the need
            for learning rate decay schedules (default: False)
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
        enable_orthograd=False,
        enable_prodigy=False,
        d0=1e-6,
        enable_schedule_free=False,
    ):
        self.torch_gc()

        if enable_prodigy and lr is None:
            lr = 1.0

        assert lr is not None and lr > 0.0
        assert d0 > 0.0
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
            enable_orthograd=enable_orthograd,
            enable_prodigy=enable_prodigy,
            d0=d0,
            enable_schedule_free=enable_schedule_free,
        )
        super(CAME, self).__init__(params, defaults)

        features = [
            enable_stochastic_rounding, enable_cautious_update,
            enable_cautious_weight_decay, enable_8bit, enable_orthograd,
            enable_prodigy, enable_schedule_free,
        ]
        if any(features):
            print("\n==== CAME Modifications ====")
            if enable_stochastic_rounding: print(f"- Stochastic Rounding enabled.")
            if enable_cautious_update: print("- Cautious Update enabled.")
            if enable_cautious_weight_decay: print("- Cautious Weight Decay enabled.")
            if enable_8bit: print(f"- 8-bit enabled: block_size={block_size}, min_quant_size={min_quant_size}.")
            if enable_orthograd: print("- Orthogonal Gradient enabled.")
            if enable_prodigy: print(f"- Prodigy enabled: d0={d0}.")
            if enable_schedule_free: print("- Schedule-Free enabled.")
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

    @staticmethod
    def apply_orthograd_(p, grad):
        eps = torch.finfo(grad.dtype).eps

        # Flatten views for operations on R^p
        theta = p.data.view(-1)
        g = grad.view(-1)

        # Capture ||∇L(θ)||
        grad_norm = g.norm(2)

        # g = ∇L(θ) - (<∇L(θ), θ> / ||θ||^2) * θ
        theta_sq_norm = torch.dot(theta, theta)

        if theta_sq_norm > eps:
            proj_factor = torch.dot(g, theta) / theta_sq_norm
            g.sub_(theta, alpha=proj_factor)

        # g_hat = (||∇L(θ)|| / (||g|| + ε)) * g
        g_orth_norm = g.norm(2)
        renorm_scale = grad_norm / (g_orth_norm + eps)
        g.mul_(renorm_scale)

        return grad

    def _prodigy_update_d(self, group):
        beta2 = group["betas"][1]
        beta2_sqrt = math.sqrt(beta2)
        d = group["prodigy_d"]
        gamma = group["lr"]

        # Scale common to all parameter contributions this step
        d2_gamma = d * d * gamma

        # Decay the group-level r scalar
        r = beta2_sqrt * group["prodigy_r"]

        # Accumulate contributions from every parameter
        s_l1_total = 0.0
        alpha_s = (1.0 - beta2_sqrt) * d2_gamma

        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if "prodigy_s" not in state:
                state["prodigy_s"] = torch.zeros_like(p.data, dtype=torch.bfloat16, device="cpu")
                state["prodigy_x0"] = p.data.detach().bfloat16().cpu().clone()

            grad = p.grad.data.float()

            # r accumulation: <g, x0 - x>  (all in fp32)
            # x0 upcasted transiently and released immediately after .item()
            x0_fp32 = state["prodigy_x0"].to(device=p.data.device, dtype=torch.float32)
            x_fp32 = p.data.float()
            inner = torch.dot(grad.view(-1), (x0_fp32 - x_fp32).view(-1)).item()
            r += alpha_s * inner
            del x0_fp32

            # s update: upcast to fp32 on the parameter device, compute, store back as bf16 on CPU
            s_fp32 = state["prodigy_s"].to(device=p.data.device, dtype=torch.float32)
            s_fp32.mul_(beta2_sqrt).add_(grad, alpha=alpha_s)
            s_l1_total += s_fp32.abs().sum().item()
            state["prodigy_s"].copy_(s_fp32.bfloat16().cpu())

        # Update group-level r
        group["prodigy_r"] = r

        # Update d estimate only when evidence is positive
        if s_l1_total > 0.0 and r > 0.0:
            d_hat = r / s_l1_total
            group["prodigy_d"] = max(d, d_hat)

    def train(self, mode: bool = True):
        for group in self.param_groups:
            if not group.get("enable_schedule_free", False):
                continue
            for p in group["params"]:
                state = self.state[p]
                if "z" in state and "x" in state:
                    if mode:
                        beta1 = group["betas"][0]
                        p.data.copy_(state["z"].lerp(state["x"], beta1))
                    else:
                        p.data.copy_(state["x"])

    def eval(self):
        self.train(False)

    @torch.inference_mode()
    def step_param(self, p, group):
        if p.grad is None:
            return

        grad = p.grad.data
        if grad.dtype in {torch.float16, torch.bfloat16}:  # TODO: keep in original dtype until float is needed?
            grad = grad.float()
        if grad.is_sparse:
            raise RuntimeError("CAME does not support sparse gradients.")

        if group["enable_orthograd"]:
            self.apply_orthograd_(p, grad)

        state = self.state[p]
        grad_shape = grad.shape
        grad_numel = grad.numel()

        factored = len(grad_shape) >= 2
        use_quantization = group["enable_8bit"] and grad_numel > group["min_quant_size"]  # TODO: add triton non_quant

        # State Initialization
        if "step" not in state:
            state["step"] = 0
            # initialize first moment with optional quantization
            if use_quantization:
                block_size = group["block_size"]
                num_blocks = (grad_numel + block_size - 1) // block_size

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
                    block_size = group["block_size"]
                    num_blocks = (grad_numel + block_size - 1) // block_size

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
        beta1, beta2, beta3 = group["betas"]
        bias_correction1 = 1.0 - beta1 ** state["step"]  # used for first moment  # TODO: make bias correction optional?
        bias_correction2 = 1.0 - beta2 ** state["step"]  # used for second moment
        bias_correction3 = 1.0 - beta3 ** state["step"]  # used for instability/confidence

        if factored:
            update = (grad**2) + group["eps"][0]

            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]

            exp_avg_sq_row.mul_(beta2).add_(
                update.mean(dim=-1), alpha=1.0 - beta2
            )
            exp_avg_sq_col.mul_(beta2).add_(
                update.mean(dim=-2), alpha=1.0 - beta2
            )

            # Approximation of exponential moving average of square of gradient
            update = _approx_sq_grad(
                exp_avg_sq_row / bias_correction2,
                exp_avg_sq_col / bias_correction2
            )
            update.mul_(grad)
        else:
            # non-factored: update second moment
            if use_quantization:
                # update = (grad**2) + group["eps"][0] is created within the kernel
                update = fused_update_exp_avg_sq_triton(
                    grad=grad,
                    exp_avg_sq_u8=state["exp_avg_sq"],
                    quant_state=state["exp_avg_sq_quant_state"],
                    beta=beta2,
                    eps=group["eps"][0],
                    bias_correction=bias_correction2,
                )
            else:
                update = (grad**2) + group["eps"][0]
                state["exp_avg_sq"].mul_(beta2).add_(
                    update, alpha=1.0 - beta2
                )
                update = state["exp_avg_sq"].div(bias_correction2).rsqrt().mul_(grad)

        # update first moment
        if use_quantization:
            rms_clip_scale = 1.0 / (update.norm(2) / (update.numel() ** 0.5) / group["clip_threshold"]).clamp(min=1.0)
            exp_avg = fused_update_exp_avg_triton(
                update=update,
                exp_avg_u8=state["exp_avg"],
                quant_state=state["exp_avg_quant_state"],
                beta=beta1,
                rms_clip_scale=rms_clip_scale,
            )
        else:
            update.div_(
                (
                    (update.norm(2) / (update.numel() ** 0.5)) / group["clip_threshold"]
                ).clamp_(min=1.0)
            )
            exp_avg = state["exp_avg"]
            exp_avg.mul_(beta1).add_(update, alpha=1 - beta1)

        if group["enable_cautious_update"]:
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg.mul_(mask)

        if factored:
            # Confidence-guided strategy
            # Calculation of instability
            res = (update - exp_avg) ** 2 + group["eps"][1]

            state["exp_avg_res_row"].mul_(beta3).add_(
                res.mean(dim=-1), alpha=1.0 - beta3
            )
            state["exp_avg_res_col"].mul_(beta3).add_(
                res.mean(dim=-2), alpha=1.0 - beta3
            )

            # Approximation of exponential moving average of instability
            res_approx = _approx_sq_grad(
                state["exp_avg_res_row"] / bias_correction3,
                state["exp_avg_res_col"] / bias_correction3
            )
            update = res_approx.mul_(exp_avg)
        else:
            update = exp_avg

        if not use_quantization:
            state["exp_avg"] = exp_avg

        if group["enable_prodigy"]:
            effective_lr = group["lr"] * group["prodigy_d"]
        else:
            effective_lr = group["lr"]

        if group["enable_schedule_free"]:
            if "z" not in state:
                state["z"] = p.data.clone()
                state["x"] = p.data.clone()
                state["lr_sq_sum"] = 0.0

            lr_sq = effective_lr ** 2
            state["lr_sq_sum"] += lr_sq

            if state["lr_sq_sum"] > 0:
                weight = lr_sq / state["lr_sq_sum"]
            else:
                weight = 1.0

            apply_update_triton(
                A=state["z"],
                B=update,
                alpha=-effective_lr,
                bias_correction=bias_correction1,
                weight_decay=group["weight_decay"],
                enable_cautious_weight_decay=group["enable_cautious_weight_decay"],
                enable_stochastic_rounding=group["enable_stochastic_rounding"],
            )

            state["x"].lerp_(state["z"], weight)

            p.data.copy_(state["z"].lerp(state["x"], beta1))
        else:
            apply_update_triton(
                A=p.data,
                B=update,
                alpha=-effective_lr,
                bias_correction=bias_correction1,
                weight_decay=group["weight_decay"],
                enable_cautious_weight_decay=group["enable_cautious_weight_decay"],
                enable_stochastic_rounding=group["enable_stochastic_rounding"],
            )

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
            if group["enable_prodigy"]:
                if "prodigy_d" not in group:
                    group["prodigy_d"] = group["d0"]
                    group["prodigy_r"] = 0.0
                self._prodigy_update_d(group)

        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p, group)

        return loss
