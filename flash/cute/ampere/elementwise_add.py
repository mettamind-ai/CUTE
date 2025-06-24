import argparse, torch, time
from typing import Type

import cuda.bindings.driver as cuda
import cutlass, cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]  # (TileM,TileN)
    blkC = gC[blk_coord]  # (TileM,TileN)
    blkCrd = cC[blk_coord]  # (TileM, TileN)

    print(f"[DSL INFO] Sliced Tensors per thread block:")
    print(f"[DSL INFO]   blkA = {blkA.type}")
    print(f"[DSL INFO]   blkB = {blkB.type}")
    print(f"[DSL INFO]   blkC = {blkC.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # # declare the atoms which will be used later for memory copy
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
    tiled_copy_B = cute.make_tiled_copy(copy_atom_load, tv_layout, tiler_mn)
    tiled_copy_C = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # allocate fragments for gmem->rmem
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    print(f"[DSL INFO] Sliced Tensors per thread:")
    print(f"[DSL INFO]   thrA = {thrA.type}")
    print(f"[DSL INFO]   thrB = {thrB.type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    for i in cutlass.range_dynamic(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    ##########################################################
    # Move data to reg address space
    ##########################################################

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = frgA.load() + frgB.load()

    # Save the results back to registers. Here we reuse b's registers.
    frgC.store(result)

    # Copy the results back to c
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):
    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mA = {mA.type}")
    print(f"[DSL INFO]   mB = {mB.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    print(f"[DSL INFO]   coord tensor = {cC.type}")

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, tv_layout, tiler_mn).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def run_elementwise_add(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    is_a_dynamic_layout=False,
    is_b_dynamic_layout=False,
    is_result_dynamic_layout=False,
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=200,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"\nRunning Elementwise Add test with:")
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    if dtype.is_integer:
        a = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randint(0, 10, (M, N), device=torch.device("cuda"), dtype=torch_dtype)
    else:
        a = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)
        b = torch.randn(M, N, device=torch.device("cuda"), dtype=torch_dtype)

    c = torch.zeros_like(a)

    print(f"Input tensor shapes:")
    print(f"a: {a.shape}, dtype: {a.dtype}")
    print(f"b: {b.shape}, dtype: {b.dtype}")
    print(f"c: {c.shape}, dtype: {c.dtype}\n")

    a_tensor = a if is_a_dynamic_layout else from_dlpack(a).mark_layout_dynamic()
    b_tensor = b if is_a_dynamic_layout else from_dlpack(b).mark_layout_dynamic()
    c_tensor = c if is_a_dynamic_layout else from_dlpack(c).mark_layout_dynamic()

    print("Compiling kernel with cute.compile ...")
    start_time = time.time()
    compiled_func = cute.compile(elementwise_add, a_tensor, b_tensor, c_tensor)
    compilation_time = time.time() - start_time
    print(f"Compilation time: {compilation_time:.4f} seconds")

    print("Executing vector add kernel...")

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()

    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    if not skip_ref_check:
        compiled_func(a_tensor, b_tensor, c_tensor)
        print("Verifying results...")
        torch.testing.assert_close(a + b, c)
        print("Results verified successfully!")

    if not benchmark: return

    # Create CUDA events for timing
    start_event = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]
    end_event   = cuda.cuEventCreate(cuda.CUevent_flags.CU_EVENT_DEFAULT)[1]

    # Warmup
    for _ in range(warmup_iterations): compiled_func(a_tensor, b_tensor, c_tensor)

    # Use the current stream for CUDA events instead of the default stream
    # Record start event
    cuda.cuEventRecord(start_event, current_stream)

    # Execute the kernel
    for _ in range(iterations): compiled_func(a_tensor, b_tensor, c_tensor)

    # Record end event
    cuda.cuEventRecord(end_event, current_stream)
    cuda.cuEventSynchronize(end_event)

    # Calculate elapsed time
    err, elapsed_time = cuda.cuEventElapsedTime(start_event, end_event)
    avg_time = elapsed_time / iterations

    # Print execution results
    print(f'''Kernel execution time: {avg_time:.4f} ms
Achieved memory throughput: {(3 * a.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9:.2f} GB/s
First few elements of result: \n{c[:3, :3]}''')

    # Destroy events
    cuda.cuEventDestroy(start_event)
    cuda.cuEventDestroy(end_event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=1024, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    run_elementwise_add(
        args.M, args.N,
        dtype=cutlass.Float32,
        is_a_dynamic_layout=True,
        is_b_dynamic_layout=True,
        is_result_dynamic_layout=True,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("\nPASS")