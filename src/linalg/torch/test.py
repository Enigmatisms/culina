import time
import torch
import cuda_reduce

def timed_print(name, func, *args, **kwargs):
    start = time.time()
    val = func(*args, **kwargs)
    interval = time.time() - start
    print(f"{name} = {val}. Time consumption: {interval * 1e6:.3f} us")


if __name__ == "__main__":
    torch.random.manual_seed(114514)

    tensor = torch.rand(131072, dtype = torch.float32, device = 'cuda')

    timed_print("Customed CUDA code (Single)", cuda_reduce.sum_cuda, tensor)
    timed_print("Customed CUDA code (Blocks)", cuda_reduce.block_sum_cuda, tensor)
    timed_print("Torch Sum", torch.sum, tensor)