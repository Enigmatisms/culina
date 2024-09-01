/**
 * @file batch_norm.cu
 * 
 * @author Qianyue He
 * @date 2024-08-26
 * 
 * Batch normalization
 * 主要做的事情：
 * （1）使用 running mean 记录训练过程中 batch 某个特征维度的均值，为什么叫 running? 因为需要 momentum
 * （2）使用 running var 记录方差
 * 使用 running mean 和 var 进行归一化（使得均值近似 0，方差近似1），训练时使用 mini batch 的均值方差归一化
 * 推理时用 running mean / var
 * （3）学习 gamma 以及 beta (反放缩，使得网络的表示能力不受影响)
 * 
 * 如果要进行并行实现，这部分：就是用 block reduce / warp reduce 就行
 */