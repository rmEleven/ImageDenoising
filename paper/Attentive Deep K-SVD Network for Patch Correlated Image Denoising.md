# 1. INTRODUCTION

1. 图像去噪是计算机视觉中的基础预处理步骤,用来处理图像中的加性高斯白噪声。

2. 过去几十年提出了大量图像去噪算法,如加权中值滤波、小波变换等。

3. 最近基于稀疏表示的分解去噪算法很受关注,如K-SVD及其变体。

4. K-SVD通过字典学习和稀疏表示实现去噪,采用OMP更新字典和表示。

5. K-SVD的深度结构能以端到端的方式学习,显示出优异的去噪性能。

6. 深度K-SVD将K-SVD扩展到深度结构,使用LISTA网络进行稀疏表示。

7. 深度K-SVD使用MLP网络学习控制稀疏性的阈值参数。

8. 当前框架将图像块作为独立样本,忽略了块间的内在相关性。

9.  图像块之间和内部存在结构和内容相关性,稀疏表达也会有相似模式。

10. 提出注意力机制增强块内外相关性,获得更准确的参数估计和稀疏表达。

11. 方法能显著提高去噪性能和收敛速度。

### 补充知识点

#### K-SVD

K-SVD是一种基于字典学习和稀疏表达的图像去噪算法,全称为K-Singular Value Decomposition。其主要思想是:

1. 将图像分割成小的重叠图像块。

2. 假设每个图像块可以在一个过完备字典上得到稀疏表达。

3. 迭代地学习一个能够更好表示图像块的字典,并得到每个块对应的稀疏系数。

4. 使用学习到的字典和稀疏系数来重构图像,从而去除噪声。

K-SVD的具体步骤是:

1. 初始化一个过完备的字典D。

2. 对每个图像块,使用 pursuit 算法(如OMP)在给定字典D上推导出稀疏表达向量。 

3. 按列更新字典D的每一个atom(原子),采用rank-1的奇异值分解方法。

4. 固定字典D,重新计算所有的稀疏表达向量。

5. 重复步骤3-4,直到达到预定迭代次数或其他停止条件。

6. 使用最后学习到的字典D和所有的稀疏表达向量,重构去噪后的图像。

K-SVD能够学习图像的全局先验知识,并用于去噪,比采用预定字典的方法效果好。它成为近年来图像去噪领域的重要方法之一。

#### DKSVD

DKSVD是K-SVD的深度版本,将K-SVD扩展到了端到端的深度网络结构。其主要改进包括:

1. 使用无监督的LISTA(Learned Iterative Shrinkage and Thresholding Algorithm)网络来学习稀疏表达,取代原来的OMP算法。

2. 学习一个多层感知机(MLP)来自适应地预测控制稀疏性的阈值参数,而不是使用固定的经验值。

3. 字典学习以监督方式进行,字典参数随网络训练迭代更新。

4. 将深度神经网络与经典算法相结合,使网络端到端可训练。

相比K-SVD,DKSVD的主要优势是:

1. 字典和稀疏表达都是可学习的,可以获得较好的非线性拟合能力。

2. 端到端学习使模型更加强大和统一。

3. 可以获得比较好的去噪效果,与其他深度学习去噪方法比较。

4. 相比其他深度去噪网络,计算复杂度较低。

DKSVD连接了经典与深度学习算法,展示了将二者结合的优势,是最近图像去噪领域的前沿技术之一。

# 2. ATTENTIVE K-SVD NETWORK

## 2.1. Sparse Coding based Image Denoising Formulation

1. 去噪目标是最小化重构图像与真实图像的误差以及所有图像块稀疏表达的L0范数。

2. 使用展开的LISTA网络提取每个图像块的稀疏系数。

3. 每个块有只一个控制稀疏性的阈值参数λk。

## 2.2. Sparsity Controller Calculation by MLP

1. 将λk扩展为向量λ^k,对每个稀疏系数学习一个控制器。

2. 构建5层的MLP网络,从噪声图像块预测λ^k。

3. 自适应学习λ^k能捕捉块的结构信息。

## 2.3. Correlation-imposing by Channel-space Attention

同一图片的图像块的系数阈值上施加了二维相关性

1. 提出含通道和空间注意力的模块。

2. 压缩通道维获得通道注意力,压缩空间维获得空间注意力。

3. 注意力作用在λ^k上获得平滑增强特征。

4. 重新整形注意力后λ^k以建立块内外相关性。

## 2.4. Patch Reconstruction

1. 重构目标是从去噪的图像补丁重构出全局的去噪图像。

2. 定义了提取图像补丁的操作符Rk。

3. 学习了一个权重参数w,用于在重构时对补丁进行加权。

4. w的大小与输入图像大小相同。

5. 重构公式为:根据Rk提取出补丁xk,进行加权然后相加,再归一化。

6. w会在网络训练过程中从数据中学习。

7. 通过加权重构,可以减少补丁之间的误差,产生更连续平滑的图像。

8. 重构是图像从补丁到全局表示的最后一步。

# 3. EXPERIMENTAL RESULTS

1. 使用BSD500数据集进行训练和评估。

2. 添加不同标准差的高斯白噪声进行去噪测试。

3. 使用Adam优化器训练,学习率设置。

4. 给出训练损失曲线,收敛迅速。

5. 在Set12和BSD68数据集上测试,与DKSVD和经典算法比较。 

6. 提出的AKSVD在PSNR和SSIM上均优于DKSVD。

7. 在BSD68上也略优于BM3D和WNNM。

8. 分析了不同的网络结构设计。

9. 给出视觉去噪效果的比较。

10. 验证了方法的有效性和显著改进。

# 4. CONCLUSIONS

1. 提出了一种基于注意力机制的深度K-SVD去噪网络。

2. 通过学习增强后的稀疏控制器,增强了图像块内外的相关性。

3. 实验结果证明,提出的方法相比原来的DKSVD,PSNR和SSIM均有显著提高。

4. 收敛速度也有显著改善。

5. 与经典算法相比也有一定的性能优势。

6. 注意力机制是提出网络的关键创新点。

7. 证明了注意力机制在图像去噪中的有效性。

8. 进一步的工作可以探索注意力机制在其他图像恢复问题中的应用。

# 特点

1. 在深度K-SVD的基础上,提出了注意力机制来增强图像块内外的相关性。

2. 学习增强后的稀疏控制器参数,而不是简单的标量阈值。

3. 注意力模块包含通道注意力和空间注意力,可学习更具区分性的特征。

4. 网络端到端可训练,字典和其他参数联合优化。

5. 显著提高了去噪效果,收敛也更加迅速。

6. 提出了一种将经典采样理论与深度学习相结合的框架。

7. 证实了注意力机制在图像去噪任务中的有效性。

8. 网络结构相对简单,计算效率较高。

9. 可以拓展到其他图像恢复问题中。

综上,AKSVD通过注意力机制建模局部结构信息,取得了比DKSVD更好的去噪效果和收敛性能,是最近去噪领域的一个有效网络结构。

# idea

1. 网络结构设计理念

- 使用监督预训练后微调到自监督任务是一个可行方向。

- 注意力机制可以有效利用数据内在结构特征。

- 网络可以结合经典方法和深度学习。

2. 训练策略

- 可以先用有标签数据预训练模型,再在无标签数据上微调。

- 可以设计合适的代理任务产生监督信号。

3. 模型分析

- 可从语义、纹理两个层面分析模型输出。

- 对比有无注意力机制的区别。

4. 评价指标

- 除PSNR/SSIM,还可以采用感知度量。

- 可进行A/B测试评价主观质量。

5. 数据增强和正则化

- 噪声、模糊、塞尔等可以作为数据增强。

- 加入先验结构特征作为正则项。