# 线性逆问题

## 1. 问题形式

$y = Ax + w$

$y, A$ 已知，求解 $x$，$w$ 是未知噪声

多数情况下A是病态的

## 2. 与图像处理的联系

### 观点1：

把 $y$ 视作观测到的退化的图像(模糊、噪声)，$A$ 视作退化过程，$x$ 视作完好的图像(求解目标)

### 观点2：

把 $y$ 视作观测到的图像，$A$ 视作一个过完备字典，$x$ 视作图像的稀疏表示

## 3. 求解方法

### 最小二乘法

求解目标：$\hat{x}_{LS} = \underset{x}{argmin} || Ax - y ||^2_2$

- 保真度较高（偏差小）
- 对扰动敏感（处理病态系统时方差大）
- 不适合求解病态方程

### 岭回归/吉洪诺夫正则化

求解目标：$\hat{x}_{T} = \underset{x}{argmin} || Ax - y ||^2_2 + \lambda || x ||^2_2$

- 权衡方差和偏差
- 适合求解病态线性系统的逆问题

# 近端梯度下降

## 1. 求解问题

$x = \underset{x}{argmin} \ g(x) + h(x)$

- $g(x)$ 是可微函数
- $f(x)$ 是不可微函数

## 2. 分类

**一般的梯度下降算法：**

$h(x) = 0$

**投影梯度下降算法：**

$$
h(x) = I_C(x) = \begin{cases}
                    0,      & x \in C \\
                    \infty, & x \notin C
                \end{cases}
$$

**迭代收缩阈值算法ISTA：**

$h(x) = \lambda||x||_1$

## 3. LASSO问题

$x = \underset{x}{argmin} \displaystyle\frac{1}{2} ||Ax-b||^2_2 + \lambda ||x||_1$

- 用L1范数替换岭回归的L2范数
- L1范数能产生稀疏解且对异常值不敏感
- 使用基于梯度的算法来求解，复杂度小，结构简单
- L0范数更直接反应稀疏度，但其优化是NP困难的

## 4. ISTA

直接令导数为0求解涉及大矩阵求逆，不实用

去掉大矩阵A后令导数为0得到软阈值函数:

$$
x = S_\lambda(b) = \begin{cases}
                        b+\lambda, & b < -\lambda \\
                        0,         & b < |\lambda| \\
                        b-\lambda, & b > \lambda \\
                    \end{cases}
$$

对g(x)做泰勒展开可以把LASSO问题转换成:

$x = \underset{x}{argmin} \displaystyle\frac{1}{2t} ||x-z||^2_2 + \lambda ||x||_1$

$z = x_0 - t \nabla g(x_0)$

$\displaystyle\frac{1}{t} = \nabla ^2 g(x_0)$

得到解:

$x = \displaystyle\frac{1}{t} \underset{x}{argmin} \displaystyle\frac{1}{2} ||x-z||^2_2 + \lambda t ||x||_1 = \displaystyle\frac{1}{t} S_{\lambda t} (z)$

定义近端算子:

$prox_{t,h(·)}(z) = \underset{x}{argmin} \displaystyle\frac{1}{2t} ||x-z||^2_2 + h(x)$

迭代过程:

1. 计算g(x)梯度并沿梯度方向更新

    $z^{(k)} = x^{(k)} - t\nabla g(x^{(k)})$

2. 进行阈值化操作

    $x^{(k+1)} = prox_{t,h(·)}(z^{(k)})$

3. 两个子问题交替迭代直到收敛


[参考资料1](https://www.cnblogs.com/louisanu/p/12045861.html)

[参考资料2](https://www.bilibili.com/video/BV1AS4y1q76x/?spm_id_from=333.999.0.0&vd_source=37fcec25f2327f81f5fcc1392e6da46c)