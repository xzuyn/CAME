import math
import gc

import numpy as np
import torch
from torch.optim import Optimizer

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True

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
        quantized_vals = quantized_vals + 0.5  # TODO: Replace with triton round or stochastic round
        
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
except ImportError:
    HAS_TRITON = False


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
        enable_cautious=False,
        enable_8bit=False,
        triton_8bit=True,
        block_size=2048,
        min_8bit_size=16384,
        quiet_8bit=True,
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
            triton_8bit=triton_8bit,
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
                print(f"- Stochastic Rounding enabled: seed={torch.initial_seed()}.")
                self.stochastic_generators = {}
                for stochastic_device in {p.device for group in self.param_groups for p in group['params']}:
                    stochastic_generator = torch.Generator(device=stochastic_device)
                    stochastic_generator.manual_seed(torch.initial_seed())
                    self.stochastic_generators[stochastic_device] = stochastic_generator
            if enable_cautious:
                print("- Cautious Masking enabled.")
            if enable_8bit:
                self.use_triton = True if HAS_TRITON and triton_8bit else False
                backend = "triton" if self.use_triton else "python"
                print(f"- 8-bit enabled: block_size={block_size}, min_8bit_size={min_8bit_size}, backend={backend}.")
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

    def _should_use_8bit(self, param_shape):
        """Determines whether parameters should be quantized to 8-bit based on size and layer type"""
        if len(param_shape) == 2:  # Linear layers
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:  # 1x1 conv
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        return False

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
    def _copy_stochastic(self, target, source):
        """
        Copies source into target using stochastic rounding
    
        Args:
            target: the target tensor with dtype=bfloat16
            source: the target tensor with dtype=float32
        """
        # create a random 16 bit integer
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
    def _add_stochastic(self, input, other, alpha=1.0):
        """
        Adds other to input using stochastic rounding

        Args:
            input: the input tensor with dtype=bfloat16
            other: the other tensor
            alpha: a multiplier for other
        """
        result = other.clone() if other.dtype == torch.float32 else other.to(dtype=torch.float32)

        result.add_(input, alpha=alpha)
        self._copy_stochastic(input, result)

    # https://github.com/NVlabs/Sana/blob/main/diffusion/utils/optimizer.py
    def _quantize_state_python(self, state_tensor, block_size):
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

    # https://github.com/NVlabs/Sana/blob/main/diffusion/utils/optimizer.py
    def _dequantize_state_python(self, quantized_chunks):
        """Dequantizes quantized chunks back to float32"""
        if not isinstance(quantized_chunks, list):
            return quantized_chunks
        chunks = []
        for c in quantized_chunks:
            chunks.append(c["data"].float() * c["scale"] + c["min"])
        return torch.cat(chunks)

    def _quantize_state_triton(self, state_tensor, block_size):
        n_elements = state_tensor.numel()
        num_quant_blocks = (n_elements + block_size - 1) // block_size
        
        mins = torch.empty((num_quant_blocks,), dtype=torch.float32, device=state_tensor.device)
        maxs = torch.empty((num_quant_blocks,), dtype=torch.float32, device=state_tensor.device)

        grid = lambda meta: (num_quant_blocks,)
        _get_block_stats_kernel[grid](state_tensor.flatten(), mins, maxs, num_quant_blocks, QUANT_BLOCK_SIZE=block_size)
        
        scales = maxs - mins
        output_data = torch.empty_like(state_tensor, dtype=torch.uint8).flatten()
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _quantize_kernel[grid](output_data, state_tensor.flatten(), scales, mins, n_elements, BLOCK_SIZE=1024, QUANT_BLOCK_SIZE=block_size)

        return output_data.reshape(state_tensor.shape), scales, mins

    def _dequantize_state_triton(self, state_data, scales, mins, original_shape, block_size):
        if state_data is None:
            return None
            
        n_elements = state_data.numel()
        output = torch.empty(original_shape, dtype=torch.float32, device=state_data.device)
        
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _dequantize_kernel[grid](output.flatten(), state_data.flatten(), scales, mins, n_elements, BLOCK_SIZE=1024, QUANT_BLOCK_SIZE=block_size)
        
        return output

    # https://github.com/NVlabs/Sana/blob/main/diffusion/utils/optimizer.py
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
            if use_8bit and self.use_triton:
                state["exp_avg"], state["exp_avg_scales"], state["exp_avg_mins"] = self._quantize_state_triton(torch.zeros_like(grad), group["block_size"])
            elif use_8bit:
                state["exp_avg"] = self._quantize_state_python(torch.zeros_like(grad), group["block_size"])
            else:
                state["exp_avg"] = torch.zeros_like(grad)

            if factored:
                state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
                state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1]).type_as(grad)
                state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).type_as(grad)
            else:
                if use_8bit and self.use_triton:
                    state["exp_avg_sq"], state["exp_avg_sq_scales"], state["exp_avg_sq_mins"] = self._quantize_state_triton(torch.zeros_like(grad), group["block_size"])
                elif use_8bit:
                    state["exp_avg_sq"] = self._quantize_state_python(torch.zeros_like(grad), group["block_size"])
                else:
                    state["exp_avg_sq"] = torch.zeros_like(grad)
            state["RMS"] = 0

        state["step"] += 1
        state["RMS"] = self._rms(p.data)

        # load / dequantize first moment
        if use_8bit and self.use_triton:
            exp_avg = self._dequantize_state_triton(state["exp_avg"], state["exp_avg_scales"], state["exp_avg_mins"], grad_shape, group["block_size"])
        elif use_8bit:
            exp_avg = self._dequantize_state_python(state["exp_avg"])
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
            if use_8bit and self.use_triton:
                exp_avg_sq = self._dequantize_state_triton(state["exp_avg_sq"], state["exp_avg_sq_scales"], state["exp_avg_sq_mins"], grad_shape, group["block_size"])
            elif use_8bit:
                exp_avg_sq = self._dequantize_state_python(state["exp_avg_sq"])
            else:
                exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
            if use_8bit and self.use_triton:
                state["exp_avg_sq"], state["exp_avg_sq_scales"], state["exp_avg_sq_mins"] = self._quantize_state_triton(exp_avg_sq, group["block_size"])
            elif use_8bit:
                state["exp_avg_sq"] = self._quantize_state_python(exp_avg_sq, group["block_size"])
            else:
                state["exp_avg_sq"] = exp_avg_sq
            update = exp_avg_sq.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # update first moment
        exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])
        # re-quantize first moment if using 8bit
        if use_8bit and self.use_triton:
            state["exp_avg"], state["exp_avg_scales"], state["exp_avg_mins"] = self._quantize_state_triton(exp_avg, group["block_size"])
        elif use_8bit:
            state["exp_avg"] = self._quantize_state_python(exp_avg, group["block_size"])
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

        if group["enable_cautious"]:
            mask = (update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            update.mul_(mask)

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
