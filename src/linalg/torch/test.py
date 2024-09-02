import tqdm
import time
import torch
import cuda_scan 
import cuda_reduce
import numpy as np
from benchmarking import *

def timed_print(name, func, *args, **kwargs):
    start = time.time()
    val = func(*args, **kwargs)
    interval = time.time() - start
    print(f"{name} = {val}. Time consumption: {interval * 1e6:.3f} us")

def comparative_test(func1, func2, args_f1, args_f2, kwargs_f1, kwargs_f2, length = 1024):
    stt_t = time.time()
    result_1 = func1(*args_f1, **kwargs_f1)
    end_t = time.time()
    print(f"Customed CUDA code takes: {(end_t - stt_t) * 1e3:.3f} ms")
    stt_t = time.time()
    result_2 = func2(*args_f2, **kwargs_f2)
    end_t = time.time()
    print(f"Torch code takes: {(end_t - stt_t) * 1e3:.3f} ms")

    diff = np.abs((result_1 - result_2).cpu().numpy())
    print(f"Max diff: {diff.max():.4f}. Mean L1 diff: {diff.mean():.4f}")
    print("First 10:")
    for i in range(10):
        print(f"{i}: ", diff[i])
    # print("Every warp:")
    # for i in range(31, length, 32):
    #     print(f"{i}: ", diff[i])

def reduce_test():
    tensor = torch.rand(131072, dtype = torch.float32, device = 'cuda')

    timed_print("Customed CUDA code (Single)", cuda_reduce.sum_cuda, tensor)
    timed_print("Customed CUDA code (Blocks)", cuda_reduce.block_sum_cuda, tensor)
    timed_print("Torch Sum", torch.sum, tensor)

def cumsum_test():
    length = 131072
    tensor = torch.rand(length, dtype = torch.float32, device = 'cuda')
    comparative_test(cuda_scan.shared_naive_cumsum, torch.cumsum, [tensor], [tensor], {}, {"dim": 0}, length)

if __name__ == "__main__":
    torch.random.manual_seed(114514)
    # cumsum_test()
    functions = [
        {'name': 'cuda: shared_naive', 'func': cuda_scan.shared_naive_cumsum, 'kwargs': {}},
        {'name': 'cuda: global_naive', 'func': cuda_scan.naive_cumsum, 'kwargs': {}},
        {'name': 'torch: cumsum', 'func': torch.cumsum, 'kwargs': {'dim': 0}},
    ]
    # one_time_print_bench_mark(
    #     functions,
    #     length = 131072,
    #     num_time = 256
    # )  
    lengths = [1024 * (2 ** i) for i in range(9)]
    x_ticks = ['$2^{%d}$'%(i + 10) for i in range(9)]
    curve_benchmark(functions, 'Parallel Prefix Sum', lengths, 40000, x_ticks = x_ticks)
    