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

### 4.3 缺陷

隐空间的分布是不连续的

潜在空间内的随机点可能给出毫无意义的结果

Autoencoder适合数据的压缩与还原，不适合生成未见过的数据。

<img src="/images/AutoencoderDisadvantage.png" width="500">

# Variational Autoencoder 变分自编码器

## 1. 相关数学知识

### 先验分布（Prior Distribution）

先验分布是指在观测新数据之前，对未知量的概率分布的预先假设或先验知识。这个分布是基于以往的经验或先前的研究得到的，通常表示为 $$P(θ)$$。$θ$ 是未知量，先验分布 $P(θ)$ 描述了在未观测数据之前，我们对未知量的不确定性有多大的估计。它为后续的推断提供了起点。

### 后验分布（Posterior Distribution）

后验分布是指在观测了新数据之后，对未知量的概率分布进行更新得到的分布。这个分布表示为 $$P(θ|D)$$，其中 $D$ 是观测到的数据。后验分布结合了先验分布和观测数据，用贝叶斯公式计算得到。它描述了在考虑了新数据后，对未知量的概率分布有多大的修正。后验分布是进行贝叶斯推断的核心结果。

### 贝叶斯公式（Bayes' Theorem）

贝叶斯公式是用于计算后验分布的重要公式，它表达了在给定观测数据的情况下，如何从先验分布得到后验分布。贝叶斯公式如下所示：
$$P(θ|D) = \frac{P(D|θ) \times P(θ)}{P(D)}$$
其中，$$P(θ|D)$$ 表示后验分布，$$P(D|θ)$$ 表示似然函数，它描述了在给定参数 $θ$ 下观测到数据 $D$ 的概率，$P(θ)$ 表示先验分布，$P(D)$ 表示边缘似然函数或边缘概率，它是观测数据的概率。在实际应用中，通常只需要计算后验分布的比例关系，而不需要计算边缘似然函数 $P(D)$ 的具体值。

### KL 散度 (Kullback-Leibler Divergence)

$KL$ 散度（Kullback-Leibler Divergence），也称为相对熵，是一种用于衡量两个概率分布之间差异的非对称度量。它用来度量当我们使用一个概率分布 $Q$ 来近似另一个概率分布 $P$ 时，由于采用了近似而产生的信息损失或不匹配程度。

$D_{KL}(P || Q) = \sum P(x) \times log(\displaystyle\frac{P(x)}{Q(x)})$

其中，$x$ 表示概率分布中的事件，$P(x)$ 和 $Q(x)$ 分别是事件 $x$ 在概率分布 $P$ 和 $Q$ 中的概率。

KL 散度具有以下性质：

1. 非负性：$D_{KL}(P || Q) ≥ 0$，当且仅当 $P$ 和 $Q$ 是相同的概率分布时，$D_{KL}$ 为零。
2. 非对称性：$D_{KL}(P || Q) ≠ D_{KL}(Q || P)$，即 $D_{KL}$ 不满足交换律。
3. 不符合三角不等式：$D_{KL}$ 不满足三角不等式。

在机器学习和优化中，$KL$ 散度常用于衡量两个概率分布之间的相似度或距离。例如，在最大似然估计和变分推断中，我们可以使用 $KL$ 散度来度量模型概率分布与真实分布之间的差异，并通过最小化 $KL$ 散度来使模型尽可能逼近真实分布。

### 变分推断 (Variational Inference, VI)

变分推断（Variational Inference，VI）是一种概率图模型的推断方法，用于近似计算复杂的后验概率分布。

在概率图模型中，我们通常面临着计算后验概率分布的困难问题，特别是在高维、复杂模型或大规模数据集的情况下，解析计算后验概率分布是不可行的。

变分推断提供了一种通过优化近似分布来近似计算后验概率分布的方法，从而实现对概率图模型进行推断和学习。

在变分推断中，我们假设一个简单的参数化分布（如高斯分布或指数族分布），用来**近似**复杂的后验概率分布。我们将这个近似分布称为变分分布（Variational Distribution）或变分后验（Variational Posterior）。然后，通过**最小化**变分分布与真实后验概率分布之间的**差异**，即 KL 散度（Kullback-Leibler Divergence），来优化变分分布的参数，使其**逼近**真实后验概率分布。

变分推断的基本思想是将原来的复杂推断问题转化为一个优化问题，通过寻找最优的变分分布参数来最小化近似误差。这样，在近似计算后验概率分布时，我们只需要处理简单的参数化分布，而避免了直接计算复杂后验分布的困难。

### 证据下界ELBO（Evidence Lower Bound）

也称为变分下界（Variational Lower Bound），是变分推断中的一个重要概念，用于衡量变分分布与真实后验概率分布之间的差异，并在变分推断中作为优化目标。

我们希望找到最优的变分分布参数，使得变分分布与真实后验概率分布尽可能接近。由于真实后验概率分布通常是难以计算的，我们无法直接得到两者的差异。

ELBO 提供了一种有效的方法来近似计算这种差异。ELBO 是一个对数似然函数（log likelihood）的下界，它表示为：

$ELBO = E[log P(X|θ)] - KL(q(θ) || P(θ))$

其中，

$E[log P(X|θ)]$ 表示数据的对数似然函数的期望，其中 $X$ 是观测数据，$θ$ 是模型参数。

$KL(q(θ) || P(θ))$ 表示变分分布 $q(θ)$ 与真实后验概率分布 $P(θ)$ 之间的 $KL$ 散度，用来衡量两个分布之间的差异。

由于 $KL$ 散度是非负的，所以 $ELBO ≤ log P(X)$。当变分分布 $q(θ)$ 完全匹配真实后验概率分布 $P(θ)$ 时，$KL$ 散度为零，此时 $ELBO$ 达到上界 $log P(X)$，即 $ELBO = log P(X)$。

在变分推断中，我们希望最大化 $ELBO$，因为这等效于最小化变分分布与真实后验概率分布之间的差异。通过最大化 $ELBO$，我们可以优化变分分布的参数，使其尽可能逼近真实后验概率分布，从而实现对复杂模型的推断和学习。

## 2. VAE

Variational Autoencoder输出的是特征的分布

<img src="/images/VariationAutoencoderAdvantage.png" width="500">

基本结构如下：

<img src="/images/VariationalAutoencoder.png" width="300">


均值 $m$ 标准差 $\sigma$ 噪声 $e$ (从正态分布获取)

最小化 $||x-\hat{x}||^2 + \sum _i (e^{\sigma _i} - (1+\sigma _i) + m^2_i)$

正则化项使 $\sigma$ 趋近1，$m$ 趋近0，即输出分布 $z$ 趋近正态分布


[参考资料1](https://www.bilibili.com/video/BV1Zq4y1h7Tu/?spm_id_from=333.999.0.0&vd_source=37fcec25f2327f81f5fcc1392e6da46c)

[参考资料2](https://www.bilibili.com/video/BV1Ge4y1S7eu/?spm_id_from=333.999.0.0&vd_source=37fcec25f2327f81f5fcc1392e6da46c)

[参考资料3](https://zhuanlan.zhihu.com/p/144649293)

[参考资料4](https://zhuanlan.zhihu.com/p/403104014)