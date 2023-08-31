# 无监督学习

## 1. 监督学习

已知：样本 + 标签
目标：找出样本到标签的映射

例子：分类、回归、目标检测、语义分割

目标检测：
<img src="/images/TargetDetection.png" width="300">

语义分割：
<img src="/images/SemanticSegmentation.png" width="300">

## 2. 无监督学习

已知：样本
目标：找出隐含在样本里的模式或结构

例子：聚类、降维、特征学习、密度估计

聚类：
<img src="/images/Clustering.png" width="300">

降维：
<img src="/images/DimensionalityReduction.png" width="300">

特征学习：
<img src="/images/FeatureLearning.png" width="300">

密度估计：
<img src="/images/DensityEstimation.png" width="300">

# 生成模型

## 1. 原理

给定训练集，产生和训练集同分布的新样本

训练样本服从 $p_{data}(x)$ 分布
产生样本服从 $p_{model}(x)$ 分布

$p_{data}(x)$ 和 $p_{model}(x)$ 相近

## 2. 应用

图像合成

图像属性编辑

图片风格转义

## 3. 分类

- 显式密度估计 explicit density
  - 定义分布并求解
  - tractable density
    - 可以求出分布
    - **PixelRNN/CNN**
  - approximate density
    - 只能求出近似分布
    - **Variational Autoencoder**
- 隐式密度估计 implicit density
  - 无需定义分布
  - direct
    - **GAN**

## 4. Autoencoder

<img src="/images/Autoencoder.png" width="300">

### 4.1 基本概念

**编码器**

进行无监督特征学习，提取数据 $x$ 中有效的低维的特征 $z$，$z$ 仅保留数据中有意义的信息

方法：CNN + ReLU

训练好后的用途：特征学习、监督分类

**解码器**

从特征 $z$ 重构输入数据 $\hat{x}$

方法：CNN + ReLU

训练好后的用途：图像生成

**训练**

调整编码器和解码器

最小化L2范数 $||x-\hat{x}||^2$

### 4.2 分类

Auto-Encoder主要可以分为以下几类:

1. 标准Auto-Encoder:最基本的Auto-Encoder结构,通过 Encoder将输入压缩到低维空间,Decoder再重构输入。

2. **稀疏Auto-Encoder(Sparse Auto-Encoder)**:在Encoder输出加入稀疏性约束,使隐层表示更为稀疏。

3. 去噪Auto-Encoder(Denoising Auto-Encoder):在输入数据加入噪声,使模型更具泛化性。

4. **变分Auto-Encoder(Variational Auto-Encoder)**:引入变分推断,使编码层输出满足标准正态分布。

5.  LSTM Auto-Encoder:使用LSTM作为Encoder和Decoder,处理顺序数据。

6. 卷积Auto-Encoder(Convolutional Auto-Encoder):使用卷积层作为编码器和解码器,处理图像等数据。

7. 防故障Encoder(Robust Auto-Encoder):训练时在输入/隐层加入干扰,增加对异常值的鲁棒性。 

8. 递归Auto-Encoder:使用递归神经网络作为自动编码器。

9. 堆叠Auto-Encoder(Stacked Auto-Encoder):串联多个Auto-Encoder形成深度非监督模型。

10. 对抗Auto-Encoder(Adversarial Auto-Encoder):加入对抗网络使隐空间正则化。

还有一些更复杂的结构,组合了上述不同类型的Auto-Encoder。不同结构各有优势,需要根据具体任务进行选择。

### 4.3 Variational Autoencoder 变分自编码器

<img src="/images/VariationalAutoencoder.png" width="300">

编码器输出的是特征的分布

均值 $m$ 标准差 $\sigma$ 噪声 $e$ (从正态分布获取)

最小化 $||x-\hat{x}||^2 + \sum _i (e^{\sigma _i} - (1+\sigma _i) + m^2_i)$

正则化项使 $\sigma$ 趋近1，$m$ 趋近0，即输出分布 $z$ 趋近正态分布


[参考资料1](https://www.bilibili.com/video/BV1Zq4y1h7Tu/?spm_id_from=333.999.0.0&vd_source=37fcec25f2327f81f5fcc1392e6da46c)