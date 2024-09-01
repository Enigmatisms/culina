import time
import torch
import cuda_reduce
import cuda_scan

def timed_print(name, func, *args, **kwargs):
    start = time.time()
    val = func(*args, **kwargs)
    interval = time.time() - start
    print(f"{name} = {val}. Time consumption: {interval * 1e6:.3f} us")

def reduce_test():
    tensor = torch.rand(131072, dtype = torch.float32, device = 'cuda')

    timed_print("Customed CUDA code (Single)", cuda_reduce.sum_cuda, tensor)
    timed_print("Customed CUDA code (Blocks)", cuda_reduce.block_sum_cuda, tensor)
    timed_print("Torch Sum", torch.sum, tensor)

def cumsum_test():
    tensor = torch.rand(16384, dtype = torch.float32, device = 'cuda')

    timed_print("Customed CUDA code (Single)", cuda_scan.naive_cumsum_long, tensor)
    timed_print("Torch Sum", torch.cumsum, tensor, dim = 0)


if __name__ == "__main__":
    torch.random.manual_seed(114514)
    cumsum_test()
    