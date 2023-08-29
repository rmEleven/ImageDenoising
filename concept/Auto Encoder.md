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

## 4. Variational Autoencoder

### 4.1 自编码器

<img src="/images/Autoencoder.png" width="300">

#### 编码器

进行无监督特征学习，提取数据 $x$ 中有效的低维的特征 $z$，$z$ 仅保留数据中有意义的信息

方法：CNN + ReLU

训练好后的用途：特征学习、监督分类

#### 解码器

从特征 $z$ 重构输入数据 $\hat{x}$

方法：CNN + ReLU

训练好后的用途：图像生成

#### 训练

调整编码器和解码器

最小化L2范数 $||x-\hat{x}||^2$

### 4.2 变分自编码器

<img src="/images/VariationalAutoencoder.png" width="300">

编码器输出的是特征的分布

均值 $m$ 标准差 $\sigma$ 噪声 $e$ (从正态分布获取)

最小化 $||x-\hat{x}||^2 + \sum _i (e^{\sigma _i} - (1+\sigma _i) + m^2_i)$

正则化项使 $\sigma$ 趋近1，$m$ 趋近0，即输出分布 $z$ 趋近正态分布


[参考资料1](https://www.bilibili.com/video/BV1Zq4y1h7Tu/?spm_id_from=333.999.0.0&vd_source=37fcec25f2327f81f5fcc1392e6da46c)