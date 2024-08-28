### CUDA GEMM


---
​	2D 大型矩阵乘法。naive 方法是很简单的，但是存在 global memory 访问过多 + uncoalesced memory 过多的问题。这里所需要实现的矩阵乘法是分块矩阵乘，包含 thread coarsening 以及 shared memory, local register 的使用

​	我们假设 coarsening 的系数是 8，也即一个线程处理 8 * 8 的区域，256 线程。

​	A * B，A 每次处理 128 行 * 16 列 （每个线程处理 8 个 float），B 取出 (16 * 128)，最后得到 128 * 128 的 patch。每个线程负责自己的 8 * 8 (一共 16 * 16 = 256 个 patch)，加至对应的输出上即可。

​	逻辑：

- 128 * 8 取块操作需要循环，一个 block 处理一整条（和列），对应了 列 / 8  或者行 / 8 （中间的 shape）
- 线程：取 A 到 shared_mem (128 行，两列 8 float)，最好是 coalesced 的形式
- 线程: 取 B 到 shared_mem (128 列，8行)，最好直接转置, global 取时取连续的 8 个但存放时竖着存（会有 L1 uncoalesced 访问，但估计还好？）
- 现在有 128 * 8 的 sa, 以及 128 * 8 的 sb，计算时：
- 根据 16 的余数与 16 的整除确定 shared memory 的访问位置
- local 开辟 8 * 8 结果存储区

​	剩下的操作就没什么难的了。输出可以直接输出到 C，因为每一块都是串行的（k 上）

​	这里剩下的优化操作就是怎么避免 bank conflict 以及怎么 double buffering 了