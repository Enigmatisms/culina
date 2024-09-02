"""
Simple Script for benchmarking
@author: Qianyue He
@date:   2024.9.2
"""

import tqdm
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

__all__ = ['one_time_print_bench_mark', 'curve_benchmark']

COLORS  = ['#B31312', '#1640D6', '#1AACAC', '#EE9322', '#2B2A4C', '#AC87C5']

def comparison(func_args):
    results = []
    for func, args, kwargs in func_args:
        stt_t1 = time.time()
        result_1 = func(*args, **kwargs)
        end_t1 = time.time()
        results.append([stt_t1, end_t1, result_1])
    return results

def one_time_print_bench_mark(funcs, length, ref_idx = -1, num_time = 512, input_gen = None):
    num_methods = len(funcs)
    timings = np.zeros(num_methods, dtype = np.float32)
    diffs   = np.zeros(num_methods, dtype = np.float32)
    if input_gen is None:
        input_gen = partial(torch.rand, dtype = torch.float32, device = 'cuda')
    for i in tqdm.tqdm(range(num_time)):
        tensor = input_gen(length)
        results = comparison([
            [funcs[j]['func'], [tensor], funcs[j]['kwargs']] for j in range(num_methods)
        ])
        ref_output = results[ref_idx][-1]
        for j, [st, et, data] in enumerate(results):
            timings[j] += et - st
            if j == ref_idx or j == num_methods + ref_idx: continue
            diffs[j] += (data - ref_output).abs().mean().cpu().item()
    diffs /= num_time
    timings /= num_time / 1e3
    print(f"Benchmarked {num_time} time(s). Problem scale: {length}")
    for i in range(num_methods):
        print(f"\nFunc: {funcs[i]['name']} avg time: {timings[i]:.3f} ms")
        if i == ref_idx or i == num_methods + ref_idx: continue 
        print(f"Average L1 difference: {diffs[i]:.7f}")

# if the functions will produce the same result, use this function
def curve_benchmark(funcs, benchmark_name, var_data, num_time = 512, input_gen = None, x_ticks = None):
    num_methods = len(funcs)
    timings = [[] for i in range(num_methods)]
    if input_gen is None:
        input_gen = partial(torch.rand, dtype = torch.float32, device = 'cuda')
    for length in var_data:
        local_tsum = [0 for _ in range(num_methods)]
        for i in tqdm.tqdm(range(num_time)):
            tensor = input_gen(length)
            results = comparison([
                [funcs[j]['func'], [tensor], funcs[j]['kwargs']] for j in range(num_methods)
            ])
            for j, [st, et, _] in enumerate(results):
                local_tsum[j] += et - st
        for i in range(num_methods):
            timings[i].append(local_tsum[i] / num_time * 1e3)
    xs = np.arange(len(var_data))
    for i in range(num_methods):
        plt.plot(xs, timings[i], label = f"{funcs[i]['name']}", color = COLORS[i])
        plt.scatter(xs, timings[i], color = COLORS[i], s = 5)
    plt.legend()
    plt.xticks(xs, var_data if x_ticks is None else x_ticks)
    plt.xlabel('Array length')
    plt.ylabel('Time consumption / ms')
    plt.title(f'Benchmark for {benchmark_name} (Problem scale: {num_time})')
    plt.grid(axis = 'both')
    plt.savefig('benchmark.jpg', dpi = 300)
