import gc
import torch
from torch.optim import Optimizer


try:
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_kernel_rhe(
        output_ptr,
        input_ptr,
        scale_ptr,
        min_ptr,
        n_elements,
        NUM_QUANT_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        vals = tl.load(input_ptr + offsets, mask=mask)

        chunk_min = tl.min(tl.where(mask, vals, float("inf")), axis=0)
        scale = (tl.max(tl.where(mask, vals, float("-inf")), axis=0) - chunk_min) / 255.0

        is_scale_zero = scale == 0.0
        vals_scaled = (vals - chunk_min) / tl.where(is_scale_zero, 1.0, scale)
        floor_val = tl.floor(vals_scaled)

        quantized_data = tl.where(
            ((vals_scaled - floor_val) == 0.5) & ((floor_val.to(tl.int32) % 2) == 0),
            floor_val,
            tl.floor(vals_scaled + 0.5),
        )
        quantized_data = tl.where(is_scale_zero, 0, quantized_data).to(tl.uint8)

        # store quantized bytes (partial store supported by mask)
        tl.store(output_ptr + offsets, quantized_data, mask=mask)

        # store per-block scale & min (only if block exists)
        if pid < NUM_QUANT_BLOCKS:
            tl.store(scale_ptr + pid, scale)
            tl.store(min_ptr + pid, chunk_min)

    @triton.jit
    def dequantize_kernel(
        output_ptr,
        data_ptr,
        scale_ptr,
        min_ptr,
        n_elements,
        QUANT_BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        quant_block_idx = offsets // QUANT_BLOCK_SIZE

        scale = tl.load(scale_ptr + quant_block_idx, mask=mask)
        min_val = tl.load(min_ptr + quant_block_idx, mask=mask)
        quantized_data = tl.load(data_ptr + offsets, mask=mask)

        dequantized_data = quantized_data.to(tl.float32) * scale + min_val

        tl.store(output_ptr + offsets, dequantized_data, mask=mask)

    @triton.jit
    def add_stochastic_kernel(
        a_ptr,
        b_ptr,
        alpha,
        seed,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        A = tl.load(a_ptr + offsets, mask=mask)
        B = tl.load(b_ptr + offsets, mask=mask)

        A = tl.cast(A, tl.float32)
        A = B + (A * alpha)
        A = tl.cast(A, tl.uint32, bitcast=True)
        A = A + (tl.randint((seed + offsets + pid).to(tl.uint32), offsets) & 0xFFFF)
        A = A & 0xFFFF0000
        A = tl.cast(A, tl.float32, bitcast=True)
        A = tl.cast(A, tl.bfloat16)

        tl.store(a_ptr + offsets, A, mask=mask)

    def quantize_state_triton_rhe(A, block_size):
        n_elements = A.numel()
        if n_elements <= 1:
            return A

        num_blocks = (n_elements + block_size - 1) // block_size
        mins = torch.empty((num_blocks,), dtype=torch.float32, device=A.device)
        scales = torch.empty((num_blocks,), dtype=torch.float32, device=A.device)
        output_data = torch.empty_like(A, dtype=torch.uint8)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        quantize_kernel_rhe[grid](
            output_data.flatten(),
            A.flatten(),
            scales,
            mins,
            n_elements,
            NUM_QUANT_BLOCKS=num_blocks,
            BLOCK_SIZE=block_size,
        )

        return output_data.reshape(A.shape), {
            "scales": scales,
            "mins": mins,
            "shape": A.shape,
            "block_size": block_size,
        }

    def dequantize_state_triton(A, quant_state):
        n_elements = A.numel()

        output = torch.empty(quant_state["shape"], dtype=torch.float32, device=A.device)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        dequantize_kernel[grid](
            output.flatten(),
            A.flatten(),
            quant_state["scales"],
            quant_state["mins"],
            n_elements,
            QUANT_BLOCK_SIZE=quant_state["block_size"],
            BLOCK_SIZE=1024,  # TODO: Tune
        )

        return output

    def add_stochastic_triton(A, B, alpha=1.0, seed=None):
        n = A.numel()
        if n == 0:
            return

        B = (
            B.clone()
            if B.dtype == torch.float32
            else B.to(dtype=torch.float32)
        )
        assert B.is_contiguous()

        if seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        add_stochastic_kernel[grid](
            A.flatten(),
            B.flatten(),
            float(alpha),
            seed,
            n,
            BLOCK_SIZE=1024,
        )

    HAS_TRITON = True
except ImportError as e:
    print(e)
    HAS_TRITON = False

try:
    from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise, quantize_nf4, dequantize_nf4
    HAS_BNB = True
except ImportError as e:
    print(e)
    HAS_BNB = False

# TODO: Add triton bnb
# TODO: Try quantize_4bit
# TODO: Try NF4, AF4, & BOF4
# TODO: Reduce the nesting if statements when choosing to quant/dequant


class CAME(Optimizer):
    """
    Implements CAME algorithm with additions.

    This implementation is based on:
      - CAME: Confidence-guided Adaptive Memory Efficient Optimization (https://arxiv.org/abs/2307.02047)
      - Revisiting BFloat16 Training (https://arxiv.org/abs/2010.06192)
      - Cautious Optimizers: Improving Training with One Line of Code (https://arxiv.org/abs/2411.16085)
      - SANA 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer
        (https://arxiv.org/abs/2501.18427)

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
        enable_4bit (bool, optional): enable 4-bit quantization for large layers (default: False)
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
        stochastic_backend="pytorch",
        enable_cautious=False,
        enable_8bit=False,
        enable_4bit=False,
        quant_backend="pytorch",
        block_size=256,
        min_quant_size=16384,
        enable_gc=False,
    ):
        self.torch_gc()

        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert stochastic_backend in ["pytorch", "triton"]
        assert quant_backend in ["pytorch", "triton", "bnb"]
        if stochastic_backend == "triton" or quant_backend == "triton":
            assert HAS_TRITON is True
        if quant_backend == "bnb":
            assert HAS_BNB is True
        assert not (enable_8bit and enable_4bit)
        assert not (enable_4bit and (quant_backend != "bnb"))

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            enable_stochastic_rounding=enable_stochastic_rounding,
            stochastic_backend=stochastic_backend,
            enable_cautious=enable_cautious,
            enable_8bit=enable_8bit,
            enable_4bit=enable_4bit,
            quant_backend=quant_backend,
            block_size=block_size,
            min_quant_size=min_quant_size,
            enable_gc=enable_gc,
        )
        super(CAME, self).__init__(params, defaults)

        print("\n==== CAME Modifications ====")
        if (
            enable_stochastic_rounding
            or enable_cautious
            or enable_8bit
            or enable_4bit
            or enable_gc
        ):
            if enable_stochastic_rounding:
                print(f"- Stochastic Rounding enabled: seed={torch.initial_seed()}, backend={stochastic_backend}.")
                self.stochastic_generators = {}
                for stochastic_device in {p.device for group in self.param_groups for p in group["params"]}:
                    stochastic_generator = torch.Generator(device=stochastic_device)
                    stochastic_generator.manual_seed(torch.initial_seed())
                    self.stochastic_generators[stochastic_device] = stochastic_generator
            if enable_cautious:
                print("- Cautious Masking enabled.")
            if enable_8bit:
                print(f"- 8-bit enabled: block_size={block_size}, min_quant_size={min_quant_size}, backend={quant_backend}.")
            if enable_4bit:
                print(f"- 4-bit enabled: block_size={block_size}, min_quant_size={min_quant_size}, backend={quant_backend}.")
            if enable_gc:
                print("- Garbage Collection enabled.")
        else:
            print("- Using original CAME implementation.")
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

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
    def _copy_stochastic_pytorch(self, target, source):
        """
        Copies source into target using stochastic rounding

        Args:
            target: the target tensor with dtype=bfloat16
            source: the target tensor with dtype=float32
        """
        # create a random 16-bit integer
        result = torch.randint(
            size=source.shape,
            device=source.device,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
            generator=self.stochastic_generators[source.device],
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))

        del result

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
    def _add_stochastic_pytorch(self, input, other, alpha=1.0):
        """
        Adds other to input using stochastic rounding

        Args:
            input: the input tensor with dtype=bfloat16
            other: the other tensor
            alpha: a multiplier for other
        """
        result = (
            other.clone()
            if other.dtype == torch.float32
            else other.to(dtype=torch.float32)
        )

        result.add_(input, alpha=alpha)
        self._copy_stochastic_pytorch(input, result)

    # Reference: https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L537C1-L563C32
    def _quantize_state_pytorch(self, A, block_size):
        n_elements = A.numel()
        if n_elements <= 1:
            return A, {}

        shape = A.shape
        num_blocks = (n_elements + block_size - 1) // block_size

        A = A.flatten()
        A = A.unsqueeze(0)
        A = torch.nn.functional.pad(A, (0, (num_blocks * block_size - n_elements)), "replicate")
        A = A.squeeze(0)
        A = A.view(num_blocks, block_size)

        block_mins = A.min(dim=1).values
        scales = (A.max(dim=1).values - block_mins) / 255.0
        is_scale_zero = scales == 0

        A = A - block_mins.unsqueeze(1)
        A = A / torch.where(is_scale_zero, 1.0, scales).unsqueeze(1)
        A = A.round()
        A = torch.where(is_scale_zero.unsqueeze(1), 0, A)
        A = A.to(torch.uint8)
        A = A.flatten()
        A = A[:n_elements]

        return A.reshape(shape), {
            "scales": scales,
            "mins": block_mins,
            "shape": shape,
            "block_size": block_size,
        }

    # Reference: https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L565C1-L582C33
    def _dequantize_state_pytorch(self, A, quant_state):
        n_elements = A.numel()
        block_size = quant_state["block_size"]

        num_blocks = (n_elements + block_size - 1) // block_size

        A = A.flatten()
        A = torch.nn.functional.pad(A, (0, (num_blocks * block_size - n_elements)), "constant", 0)
        A = A.view(num_blocks, block_size)
        A = A.float()
        A = A * quant_state["scales"].unsqueeze(1)
        A = A + quant_state["mins"].unsqueeze(1)
        A = A.flatten()
        A = A[:n_elements]

        return A.reshape(quant_state["shape"])

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
        use_quantization = (group["enable_8bit"] or group["enable_4bit"]) and self._should_use_quantization(grad_shape)

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            # initialize first moment with optional quantization
            if use_quantization:
                if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                    state["exp_avg"], state["exp_avg_quant_state"] = self._quantize_state_pytorch(
                        torch.zeros_like(grad), group["block_size"]
                    )
                elif group["enable_8bit"] and group["quant_backend"] == "triton":
                    state["exp_avg"], state["exp_avg_quant_state"] = quantize_state_triton_rhe(
                        torch.zeros_like(grad), group["block_size"]
                    )
                elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                    state["exp_avg"], state["exp_avg_quant_state"] = quantize_blockwise(
                        torch.zeros_like(grad), blocksize=group["block_size"]
                    )
                elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                    state["exp_avg"], state["exp_avg_quant_state"] = quantize_nf4(
                        torch.zeros_like(grad), blocksize=group["block_size"]
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
                    if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                        state["exp_avg_sq"], state["exp_avg_quant_state"] = self._quantize_state_pytorch(
                            torch.zeros_like(grad), group["block_size"]
                        )
                    elif group["enable_8bit"] and group["quant_backend"] == "triton":
                        state["exp_avg_sq"], state["exp_avg_quant_state"] = quantize_state_triton_rhe(
                            torch.zeros_like(grad), group["block_size"]
                        )
                    elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                        state["exp_avg_sq"], state["exp_avg_quant_state"] = quantize_blockwise(
                            torch.zeros_like(grad), blocksize=group["block_size"]
                        )
                    elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                        state["exp_avg_sq"], state["exp_avg_quant_state"] = quantize_nf4(
                            torch.zeros_like(grad), blocksize=group["block_size"]
                        )
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            state["RMS"] = 0

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        # load / dequantize first moment
        if use_quantization:
            if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                exp_avg = self._dequantize_state_pytorch(state["exp_avg"], state["exp_avg_quant_state"])
            elif group["enable_8bit"] and group["quant_backend"] == "triton":
                exp_avg = dequantize_state_triton(state["exp_avg"], state["exp_avg_quant_state"])
            elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                exp_avg = dequantize_blockwise(
                    state["exp_avg"], quant_state=state["exp_avg_quant_state"], blocksize=group["block_size"]
                )
            elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                exp_avg = dequantize_nf4(
                    state["exp_avg"], quant_state=state["exp_avg_quant_state"], blocksize=group["block_size"]
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
                if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                    exp_avg_sq = self._dequantize_state_pytorch(state["exp_avg_sq"], state["exp_avg_sq_quant_state"])
                elif group["enable_8bit"] and group["quant_backend"] == "triton":
                    exp_avg_sq = dequantize_state_triton(state["exp_avg_sq"], state["exp_avg_sq_quant_state"])
                elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                    exp_avg_sq = dequantize_blockwise(
                        state["exp_avg_sq"], quant_state=state["exp_avg_sq_quant_state"], blocksize=group["block_size"]
                    )
                elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                    exp_avg_sq = dequantize_nf4(
                        state["exp_avg_sq"], quant_state=state["exp_avg_sq_quant_state"], blocksize=group["block_size"]
                    )
            else:
                exp_avg_sq = state["exp_avg_sq"]

            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])

            if use_quantization:
                if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                    state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = self._quantize_state_pytorch(
                        exp_avg_sq, group["block_size"]
                    )
                elif group["enable_8bit"] and group["quant_backend"] == "triton":
                    state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = quantize_state_triton_rhe(
                        exp_avg_sq, group["block_size"]
                    )
                elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                    state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = quantize_blockwise(
                        exp_avg_sq, blocksize=group["block_size"]
                    )
                elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                    state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = quantize_nf4(
                        exp_avg_sq, blocksize=group["block_size"]
                    )
            else:
                state["exp_avg_sq"] = exp_avg_sq

            update = exp_avg_sq.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # update first moment
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

        # re-quantize first moment if using quantization
        if use_quantization:
            if group["enable_8bit"] and group["quant_backend"] == "pytorch":
                state["exp_avg"], state["exp_avg_quant_state"] = self._quantize_state_pytorch(
                    exp_avg, group["block_size"]
                )
            elif group["enable_8bit"] and group["quant_backend"] == "triton":
                state["exp_avg"], state["exp_avg_quant_state"] = quantize_state_triton_rhe(exp_avg, group["block_size"])
            elif group["enable_8bit"] and group["quant_backend"] == "bnb":
                state["exp_avg"], state["exp_avg_quant_state"] = quantize_blockwise(exp_avg, blocksize=group["block_size"])
            elif group["enable_4bit"] and group["quant_backend"] == "bnb":
                state["exp_avg"], state["exp_avg_quant_state"] = quantize_nf4(exp_avg, blocksize=group["block_size"])
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
                if group["stochastic_backend"] == "triton":
                    add_stochastic_triton(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
                else:
                    self._add_stochastic_pytorch(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
            if group["stochastic_backend"] == "triton":
                add_stochastic_triton(p.data, -update)
            else:
                self._add_stochastic_pytorch(p.data, -update)
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
