# TODO: Compare python and triton 8bit methods to bitsandbytes blockwise
# TODO: Try quantize_4bit

import gc
import numpy as np
import torch
from torch.optim import Optimizer


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _get_block_stats_kernel(
        input_ptr,
        min_ptr,
        max_ptr,
        num_quant_blocks,
        QUANT_BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel to find the min and max for each block in parallel."""
        pid = tl.program_id(axis=0)
        if pid >= num_quant_blocks:
            return

        block_start = pid * QUANT_BLOCK_SIZE
        offsets = block_start + tl.arange(0, QUANT_BLOCK_SIZE)

        block_vals = tl.load(input_ptr + offsets, eviction_policy="evict_first")

        block_min = tl.min(block_vals, axis=0)
        block_max = tl.max(block_vals, axis=0)

        tl.store(min_ptr + pid, block_min)
        tl.store(max_ptr + pid, block_max)

    @triton.jit
    def _quantize_kernel(
        output_ptr,
        input_ptr,
        scale_ptr,
        min_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        QUANT_BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel to quantize a tensor using pre-computed stats."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        quant_block_idx = offsets // QUANT_BLOCK_SIZE

        scale = tl.load(scale_ptr + quant_block_idx, mask=mask)
        min_val = tl.load(min_ptr + quant_block_idx, mask=mask)

        input_vals = tl.load(input_ptr + offsets, mask=mask)

        quantized_vals = ((input_vals - min_val) / scale) * 255.0

        # Compute floor and fractional part
        q_floor = tl.floor(quantized_vals)
        frac = quantized_vals - q_floor

        # Simple per-element LCG RNG (uint32)
        # seed derived from offsets and program id so different threads get different streams
        seed = (offsets + pid * 196314165).to(tl.uint32)

        # LCG constants (use Python ints, cast result to uint32)
        a = 1664525
        c = 1013904223
        state = (seed * a + c).to(tl.uint32)

        # convert to float in [0,1)
        rnd = state.to(tl.float32) / 4294967296.0

        # increment with probability = fractional part
        inc = tl.where(rnd < frac, 1.0, 0.0)

        quantized_vals = q_floor + inc

        quantized_vals = tl.where(quantized_vals > 255.0, 255.0, quantized_vals)
        quantized_vals = tl.where(quantized_vals < 0.0, 0.0, quantized_vals)
        quantized_vals = quantized_vals.to(tl.uint8)

        tl.store(output_ptr + offsets, quantized_vals, mask=mask)

    @triton.jit
    def _dequantize_kernel(
        output_ptr,
        data_ptr,
        scale_ptr,
        min_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        QUANT_BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel to dequantize a tensor."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        quant_block_idx = offsets // QUANT_BLOCK_SIZE

        scale = tl.load(scale_ptr + quant_block_idx, mask=mask)
        min_val = tl.load(min_ptr + quant_block_idx, mask=mask)

        quantized_data = tl.load(data_ptr + offsets, mask=mask)

        dequantized_data = (quantized_data.to(tl.float32) / 255.0) * scale + min_val

        tl.store(output_ptr + offsets, dequantized_data, mask=mask)

    @triton.jit
    def _copy_stochastic_kernel(
        target_ptr,           # pointer to bfloat16 output
        source_ptr,           # pointer to float32 input
        seed,                 # scalar seed
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Kernel that implements stochastic rounding of float32 'source' into bfloat16 'target'.
        The strategy mirrors the CPU implementation: bitcast float32 -> uint32, add a uniform
        16-bit random integer to the low 16 bits, mask-off the low 16 bits, bitcast back to float32,
        then store as bfloat16.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # load source (float32)
        src = tl.load(source_ptr + offs, mask=mask)

        # bitcast float32 -> uint32
        src_bits = tl.cast(src, tl.uint32, bitcast=True)

        # generate random int32 per element and reduce to 16 bits
        rnd = tl.randint(seed, offs)
        rnd16 = rnd & 0xFFFF

        # add rnd to low bits and clear low 16 bits to simulate stochastic rounding
        added = src_bits + rnd16
        rounded_bits = added & 0xFFFF0000

        # bitcast back to float32
        rounded = tl.cast(rounded_bits, tl.float32, bitcast=True)

        # store as bfloat16 (numeric cast)
        out = tl.cast(rounded, tl.bfloat16)
        tl.store(target_ptr + offs, out, mask=mask)

    @triton.jit
    def _add_stochastic_kernel(
        input_ptr,            # pointer to bfloat16 in-place tensor (will be updated)
        other_ptr,            # pointer to float32 'other'
        alpha,                # float32 scalar multiplier
        seed,                 # scalar seed
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Kernel that computes `other + alpha * input` (where input is bfloat16), then stochastically rounds
        the result back into bfloat16 and stores it into input_ptr in-place.
        Uses the same bit-level stochastic rounding approach as _copy_stochastic_kernel.
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        # load input (bfloat16) and other (float32)
        inp = tl.load(input_ptr + offs, mask=mask)
        other = tl.load(other_ptr + offs, mask=mask)

        # cast input to float32 to compute sum
        inp_f = tl.cast(inp, tl.float32)
        sum_val = other + inp_f * alpha

        # bitcast sum_val to uint32
        sum_bits = tl.cast(sum_val, tl.uint32, bitcast=True)

        # random 16-bit int
        rnd = tl.randint(seed, offs)
        rnd16 = rnd & 0xFFFF

        # add and clear low 16 bits
        added = sum_bits + rnd16
        rounded_bits = added & 0xFFFF0000

        # bitcast back to float32
        rounded = tl.cast(rounded_bits, tl.float32, bitcast=True)

        # store back as bfloat16
        out = tl.cast(rounded, tl.bfloat16)
        tl.store(input_ptr + offs, out, mask=mask)
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    if HAS_TRITON:
        from bitsandbytes.triton.ops import  quantize_blockwise, dequantize_blockwise
    else:
        from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


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
        block_size (int, optional): quantization block size for 8-bit (default: 2048)
        min_8bit_size (int, optional): minimum number of parameters to use 8-bit (default: 16384)
        quiet_8bit (bool, optional): don't print layer info (default: True)
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
        stochastic_backend="python",
        enable_cautious=False,
        enable_8bit=False,
        quant_backend="python",
        block_size=256,
        min_8bit_size=16384,
        quiet_8bit=True,
        enable_gc=False,
    ):
        self.torch_gc()

        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert stochastic_backend in ["python", "triton"]
        assert quant_backend in ["python", "triton", "bnb", "triton_bnb"]
        if stochastic_backend == "triton":
            assert HAS_TRITON is True
        if quant_backend in ["triton", "triton_bnb"]:
            assert HAS_TRITON is True
        if quant_backend in ["bnb", "triton_bnb"]:
            assert HAS_BNB is True

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
            quant_backend=quant_backend,
            block_size=block_size,
            min_8bit_size=min_8bit_size,
            quiet_8bit=quiet_8bit,
            enable_gc=enable_gc,
        )
        super(CAME, self).__init__(params, defaults)

        print("\n==== CAME Modifications ====")
        if (
            enable_stochastic_rounding
            or enable_cautious
            or enable_8bit
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
                print(f"- 8-bit enabled: block_size={block_size}, min_8bit_size={min_8bit_size}, backend={quant_backend}.")
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
    def _should_use_8bit(self, param_shape):
        """Determine if a parameter should be quantized to 8bit

        Rules:
        1. linear layers: parameter size > min_8bit_size
        2. 1x1 conv layers: parameter size > min_8bit_size
        3. other layers: use 32bit
        """
        if len(param_shape) == 2:  # linear layer
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        return False  # other layers are not quantized

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

    # https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py
    def _copy_stochastic_python(self, target, source):
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
    def _add_stochastic_python(self, input, other, alpha=1.0):
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
        self._copy_stochastic_python(input, result)

    def _copy_stochastic_triton(self, target, source):
        n = source.numel()
        if n == 0:
            return

        t_copy_stochastic_seed = torch.randint(
            low=0,
            high=2 ** 31,
            size=(1,),
            device=source.device,
            generator=self.stochastic_generators[source.device]
        )

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _copy_stochastic_kernel[grid](
            target.flatten(),
            source.flatten(),
            int(t_copy_stochastic_seed.item()),
            n,
            BLOCK_SIZE=1024,
        )

    def _add_stochastic_triton(self, input, other, alpha=1.0):
        n = input.numel()
        if n == 0:
            return

        t_add_stochastic_seed = torch.randint(
            low=0,
            high=2 ** 31,
            size=(1,),
            device=input.device,
            generator=self.stochastic_generators[input.device]
        )

        # ensure other is float32 contiguous
        if other.dtype != torch.float32:
            other_f = other.to(dtype=torch.float32)
        else:
            other_f = other

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _add_stochastic_kernel[grid](
            input.flatten(),
            other_f.flatten(),
            float(alpha),
            int(t_add_stochastic_seed.item()),
            n,
            BLOCK_SIZE=1024,
        )

    # https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L537C1-L563C32
    def _quantize_state_python(self, state_tensor, block_size):
        """Quantize a state tensor to 8bit

        Args:
            state_tensor: tensor to be quantized
            block_size: quantization block size

        Returns:
            list of quantized data blocks, each block contains:
            - data: uint8 data
            - scale: quantization scale
            - min: minimum value
        """
        if state_tensor.numel() <= 1:
            return state_tensor

        quantized_chunks = []
        for chunk in state_tensor.split(block_size):
            # Calculate quantization parameters
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            scale = (chunk_max - chunk_min) / 255

            # Quantize to 0-255 range
            quantized_chunks.append({"data": ((chunk - chunk_min) / scale).round().byte(), "scale": scale, "min": chunk_min})
        return quantized_chunks

    # https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L565C1-L582C33
    def _dequantize_state_python(self, quantized_chunks):
        """Dequantize 8bit quantized data to 32bit float

        Args:
            quantized_chunks: list of quantized data blocks

        Returns:
            dequantized 32bit float tensor
        """
        if not isinstance(quantized_chunks, list):
            return quantized_chunks

        chunks = []
        for chunk_dict in quantized_chunks:
            # Dequantize: value = data * scale + min
            chunks.append(chunk_dict["data"].float() * chunk_dict["scale"] + chunk_dict["min"])
        return torch.cat(chunks)

    def _quantize_state_triton(self, state_tensor, block_size):
        n_elements = state_tensor.numel()
        num_quant_blocks = (n_elements + block_size - 1) // block_size

        mins = torch.empty((num_quant_blocks,), dtype=torch.float32, device=state_tensor.device)
        maxs = torch.empty((num_quant_blocks,), dtype=torch.float32, device=state_tensor.device)

        grid = lambda meta: (num_quant_blocks,)
        _get_block_stats_kernel[grid](
            state_tensor.flatten(),
            mins,
            maxs,
            num_quant_blocks,
            QUANT_BLOCK_SIZE=block_size,
        )

        scales = maxs - mins
        output_data = torch.empty_like(state_tensor, dtype=torch.uint8).flatten()

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _quantize_kernel[grid](
            output_data,
            state_tensor.flatten(),
            scales,
            mins,
            n_elements,
            BLOCK_SIZE=1024,
            QUANT_BLOCK_SIZE=block_size,
        )

        return output_data.reshape(state_tensor.shape), scales, mins

    def _dequantize_state_triton(self, state_data, scales, mins, original_shape, block_size):
        if state_data is None:
            return None

        n_elements = state_data.numel()
        output = torch.empty(original_shape, dtype=torch.float32, device=state_data.device)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _dequantize_kernel[grid](
            output.flatten(),
            state_data.flatten(),
            scales,
            mins,
            n_elements,
            BLOCK_SIZE=1024,
            QUANT_BLOCK_SIZE=block_size,
        )

        return output

    # https://github.com/NVlabs/Sana/blob/3fed41f52a5300c3063068b5f7c5dfffb4fd0f3e/diffusion/utils/optimizer.py#L501C1-L521C97
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
        use_8bit = group["enable_8bit"] and self._should_use_8bit(grad_shape)

        # State Initialization
        if len(state) == 0:
            state["step"] = 0
            # initialize first moment with optional 8-bit quantization
            if not group["quiet_8bit"]:
                self.print_layer_info(grad_shape, use_8bit)
            if use_8bit:
                if group["quant_backend"] in ["bnb", "triton_bnb"]:
                    state["exp_avg"], state["exp_avg_quant_state"] = quantize_blockwise(torch.zeros_like(grad), blocksize=group["block_size"])
                elif group["quant_backend"] == "triton":
                    (
                        state["exp_avg"],
                        state["exp_avg_scales"],
                        state["exp_avg_mins"],
                    ) = self._quantize_state_triton(torch.zeros_like(grad), group["block_size"])
                else:
                    state["exp_avg"] = self._quantize_state_python(torch.zeros_like(grad), group["block_size"])
            else:
                state["exp_avg"] = torch.zeros_like(grad)

            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
            else:
                if use_8bit:
                    if group["quant_backend"] in ["bnb", "triton_bnb"]:
                        state["exp_avg_sq"], state["exp_avg_quant_state"] = quantize_blockwise(torch.zeros_like(grad), blocksize=group["block_size"])
                    elif group["quant_backend"] == "triton":
                        (
                            state["exp_avg_sq"],
                            state["exp_avg_sq_scales"],
                            state["exp_avg_sq_mins"],
                        ) = self._quantize_state_triton(torch.zeros_like(grad), group["block_size"])
                    else:
                        state["exp_avg_sq"] = self._quantize_state_python(torch.zeros_like(grad), group["block_size"])
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            state["RMS"] = 0

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        # load / dequantize first moment
        if use_8bit:
            if group["quant_backend"] in ["bnb", "triton_bnb"]:
                exp_avg = dequantize_blockwise(state["exp_avg"], quant_state=state["exp_avg_quant_state"], blocksize=group["block_size"])
            elif group["quant_backend"] == "triton":
                exp_avg = self._dequantize_state_triton(
                    state["exp_avg"],
                    state["exp_avg_scales"],
                    state["exp_avg_mins"],
                    grad_shape,
                    group["block_size"],
                )
            else:
                exp_avg = self._dequantize_state_python(state["exp_avg"])
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
            if use_8bit:
                if group["quant_backend"] in ["bnb", "triton_bnb"]:
                    exp_avg_sq = dequantize_blockwise(state["exp_avg_sq"], quant_state=state["exp_avg_quant_state"], blocksize=group["block_size"])
                elif group["quant_backend"] == "triton":
                    exp_avg_sq = self._dequantize_state_triton(
                        state["exp_avg_sq"],
                        state["exp_avg_sq_scales"],
                        state["exp_avg_sq_mins"],
                        grad_shape,
                        group["block_size"],
                    )
                else:
                    exp_avg_sq = self._dequantize_state_python(state["exp_avg_sq"])
            else:
                exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
            if use_8bit:
                if group["quant_backend"] in ["bnb", "triton_bnb"]:
                    state["exp_avg_sq"], state["exp_avg_sq_quant_state"] = quantize_blockwise(exp_avg_sq, blocksize=group["block_size"])
                elif group["quant_backend"] == "triton":
                    (
                        state["exp_avg_sq"],
                        state["exp_avg_sq_scales"],
                        state["exp_avg_sq_mins"],
                    ) = self._quantize_state_triton(exp_avg_sq, group["block_size"])
                else:
                    state["exp_avg_sq"] = self._quantize_state_python(exp_avg_sq, group["block_size"])
            else:
                state["exp_avg_sq"] = exp_avg_sq
            update = exp_avg_sq.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # update first moment
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])
        # re-quantize first moment if using 8bit
        if use_8bit:
            if group["quant_backend"] in ["bnb", "triton_bnb"]:
                state["exp_avg"], state["exp_avg_quant_state"] = quantize_blockwise(exp_avg, blocksize=group["block_size"])
            elif group["quant_backend"] == "triton":
                (
                    state["exp_avg"],
                    state["exp_avg_scales"],
                    state["exp_avg_mins"],
                ) = self._quantize_state_triton(exp_avg, group["block_size"])
            else:
                state["exp_avg"] = self._quantize_state_python(exp_avg, group["block_size"])
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
                if group["stochastic_backend"] == "triton" and p.numel() >= 16_777_216:
                    self._add_stochastic_triton(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
                else:
                    self._add_stochastic_python(p.data, p.data, alpha=-group["weight_decay"] * group["lr"])
            else:
                p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        update.mul_(group["lr"])
        if p.dtype == torch.bfloat16 and group["enable_stochastic_rounding"]:
            if group["stochastic_backend"] == "triton" and p.numel() >= 16_777_216:
                self._add_stochastic_triton(p.data, -update)
            else:
                self._add_stochastic_python(p.data, -update)
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
