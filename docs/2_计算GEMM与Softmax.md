# 计算优化总结

## GEMM

### 基础版本

本文通过mma的m16n8k16完成matmul。下图是一个MMA的操作示意图，可以看出：
* A的维度为(m, k) = (16, 16)
* B的维度为(n, k) = (8, 16)
* C和D的维度为(m, n) = (16, 8)

![alt text](image.png)

那么如果要完成[数据访问与Swizzling](./1_数据访问与Swizzling.md)中，从smem搬运到RF上的数据，对应的迭代次数如下：

|     | A         | A Shape (Registers) | B               | B Shape (Registers) | Iteration Shape (k, m, n) |
|-----|-----------|---------------------|-----------------|---------------------|---------------------------|
|     | Q         | (2, 16)             | KT              | (8, 16)             | (8, 1, 8)                 |
|     | P         | (2, 8)              | Vj (transposed) | (16, 8)             | (4, 1, 16)                |

### 沿着k维度切分计算任务

上面的版本是一个最基础的版本，但存在几个问题：

1. ldmatrix单次拷贝(16, 16)矩阵的元素，然后迭代着依次拷贝下去。在拷贝后面的元素的时候，前面的元素已经准备好了，可以开始计算；
2. 仔细观察reserve的A shape和B shape，k-dimension很大，对应的是d_head维度。而当B_r或者B_c增加的时候，m和n维度会变大，从而很容易导致register spill。

此处引入cutlass GEMM关于k维度的优化策略。

#### cutlass GEMM: spliced-k, split-k & stream-k

cutlass GEMM关于K维度的优化策略，主要是针对m/n维度较小而k维度较大的GEMM。

基本策略是：对于一个大小维`[BM, BN, BK]`维度的work tile，除了沿着M/N维度切分之外，沿着K维度也把任务进行了切分。如此一来，待处理的可并行任务的工作量从`[WM, WN, BK]`变成了`[WM, WN, WK]`，与之相对应的，并行处理任务数量从`[BM/WM, BN/WN, BK]`变成了`[BM/WM, BN/WN, BK/WK]`。

由于沿着K维度做了任务的切分，每个执行子单元算出来的值是不完整的，需要和相同m/n沿着k维度切分的其他fragments的计算结果做reduce。

![alt text](image-36.png)


这里显示出了spliced-k和split-k由于执行单元的并行化粒度，导致的reduce阶段的复杂性的差异。下面是两者的定义：

sliced-k: reduction across warps on shared memory in CTA
split-K: reduction across CTAs

总体来说，sliced-k发生在一个SM上，因此可以通过shared memory进行结果的同步，比如依靠shared memory的一个partial accumulation sums累加结果；而split-K由于设计到跨SM的数据同步，有一些更复杂的规约策略。

细节请参考[cutlass GEMM——sliced-K、split-K & stream-K 分析 （一）](https://zhuanlan.zhihu.com/p/713411778)


#### 应用在本案中

本案的任务相对而言更直观，是在一个warp内部进行reduce的，所以可以复用MMA的C&D的寄存器累加结果，完成在RF上的规约。

由此，Q, Kj和Vj的RF shape可以进一步削减为下表。Q, Kj 和Vj的每个slice包含2个fragments，也就是16个元素的宽度，这也是`ldmatrix`的加载宽度。

| Tensor | Format      | Full Shape | Sub-tile Shape (Fragments) | # Tiles   | mma matrix variable (Fragments) |
|--------|-------------|------------|----------------|-----------|---------------------------------|
| Q      | Row major   | (2, 16)    | (2, 2)         | (1, 8)    | A                               |
| Kj     | Row major   | (8, 16)    | (8, 2)         | (1, 8)    | B                               |
| Vj     | Column major| (16, 8)    | (16, 2)        | (1, 4)    | B                               |


* 为什么选择在k维度上切分任务
  
  上述cutlass的GEMM优化策略固然好，但我们的case里，MMA是在一个warp上完成的，沿着A/B维度切分，保留完整的K维度，计算的结果也即最终结果，不是很好吗？

  这里还可以通过算存比进行定量分析，假设A有`Fr = 2`个MMA的行fragment，B有`Fc = 8`个MMA的列fragment，那么：
  
  1. 如下图，按行加载A，按列加载B。为了计算GEMM，A的每行计算，都需要重新加载一遍完整的B：`MMAs performed / Total fragments loads = Fr * Fc / (Fr + Fr * Fc) = 0.89`
    ![alt text](image-37.png)

  2. 如下图，按照k维度切分，每次加载的行和列都被充分利用了，没有B矩阵的重复加载：`MMAs performed / Total fragments loads = Fr * Fc / (Fr + Fc) = 1.6`
    ![alt text](image-38.png)

#### Double buffering

根据上一节，一个GEMM沿着K维度被做了切分，切分宽度为`ldmatrix`的加载宽度，即16个元素。一个优化策略是做一个SMEM -> RF的double buffering，从而减少MMA的数据准备时间。

![alt text](image-39.png)

#### m16n8k16对B矩阵的形状要求 & K矩阵的搬运

在[数据访问与Swizzling](./1_数据访问与Swizzling.md)中介绍过，Q, K和V都是使用`ldmatrix.x4`进行加载的。

而使用`ldmatrix.x4`时，加载的(2, 2)维的(8, 8) fragment，是按照col-major排列的，比如thread 0对A矩阵的访问顺序为(0, 1) -> (128, 129) -> (8, 9) -> (136, 137)。而做MMA计算时，对于thread 0，针对B矩阵的元素访问顺序是(0, 1) -> (8, 9)，完成计算需要依次访问(0, 1)和(8, 9)这4个half元素存储的两个32-bit寄存器。

![alt text](image-5.png)
![alt text](image-9.png)

可以看出，由于通过`ldmatrix.x4`加载B矩阵，导致(0, 1) 和(8, 9) 不连续了。这里造成Q矩阵的形状与MMA的要求有了轻微的差异，从而造成编译器需要分配额外的instruction，整理Q矩阵。

解决方案：在使用`ldmatrix.x4`加载Q矩阵时，交换(1, 0)和(0, 1) fragment的位置，这样就使得(0, 1) 和(8, 9) 按列存储，从而使得thread 0可以连续访问。

![alt text](image-40.png)

* 与V矩阵加载的比较

  这里与V矩阵从smem到RF的加载做个比较：同样是交换了(1, 0)和(0, 1) fragment的位置，在smem按row-major存储的K矩阵，由于计算的`S = Q @ K^T`要求K矩阵是转置的，而MMA要求B矩阵为col-major，当通过`ldmatrix`加载后，MMA按col-major的顺序读取row-major存储的(8, 8) fragment，相当于对K矩阵做了转置；而smem按照row-major存储的V矩阵，需要完成的计算是`O = P @ V`，要求V是非转置的，因此需要通过`ldmatrix.trans`加载(8, 8) V的fragment。

