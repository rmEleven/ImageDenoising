# 0. 总结

1. 提出了一种名为A-DLISTA的压缩感知模型,可以处理不同的测量设置(每个样本不同的测量矩阵)并适应当前数据实例。论文从理论上解释了这种设计的动机,并通过实验表明它可以优于其他LISTA类模型。

2. 提出了一种名为VLISTA的变分学习ISTA模型,可以学习稀疏字典上的分布。该模型可以看作是Bayesian LISTA模型,其中A-DLISTA作为似然模型。

3. VLISTA可以随着优化动态适应字典。因此它可以看作是一个分层表示学习方法,其中字典原子逐步允许更精细的信号恢复。

4. 字典分布可以用于检测异常分布(OOD)。VLISTA可以评估重建信号上的不确定性,这对于不需要访问真值数据的异常检测非常有用。

5. 在多个数据集上实验表明,A-DLISTA优于ISTA和LISTA等模型;VLISTA优于Bayesian压缩感知等Bayes模型,尤其是在测量数较少时。在检测OOD方面,VLISTA也优于Bayesian压缩感知。

6. 综上,该论文通过变分Bayes框架成功地解决了联合字典学习和稀疏表示学习问题,并具有很好的适应性和检测OOD的能力。

根据论文的内容,我来概括论文中提出的两个网络A-DLISTA和VLISTA的结构以及工作流程:

## 0.1 A-DLISTA网络

网络结构:

- 包含soft-threshold层和augmentation层

- soft-threshold层类似ISTA迭代,学习字典 $Ψ_t$

- augmentation层从 $Φ$ 和 $Ψ_t$ 生成自适应参数 $γ_t$ 、$θ_t$

工作流程:

- 输入:观测 $y$ ,测量矩阵 $Φ$

- soft-threshold层进行稀疏编码,用 $Ψ_t$ 重构信号

- augmentation层根据 $Φ$ 和 $Ψ_t$ 输出 $γ_t$ 、$θ_t$ 给soft-threshold层

- soft-threshold层使用 $γ_t$ 、$θ_t$ 进行编码

- 循环此过程完成信号重建

## 0.2 VLISTA网络

网络结构:

- 包含三部分:先验网络、后验网络和似然网络

- 先验网络:生成字典的先验高斯分布参数

- 后验网络:生成字典的后验高斯分布参数 

- 似然网络:基于A-DLISTA,输出信号分布参数

工作流程:

1) 根据先验网络生成初始字典先验分布 $p(Ψ_1)$

2) 根据先验采样出字典 $Ψ_1$

3) 将 $Ψ_1$ 输入似然网络A-DLISTA,得到信号分布 $p(x_1|Ψ_1)$

4) 根据 $x_1$ 和观测,后验网络输出字典的后验分布 $q(Ψ_2|x_1)$

5) 根据后验 $q(Ψ_2|x_1)$ 采样出字典 $Ψ_2$

6) 将 $Ψ_2$ 输入A-DLISTA,得到信号分布 $p(x_2|Ψ_1,Ψ_2)$

7) 根据 $x_2$ 和观测,后验网络输出 $q(Ψ_3|x_2)$

8) 重复3-7步,进行变分Bayes学习

综上所述,两种网络的结构和工作流程都是围绕字典学习和信号重建展开的,通过不同的方式实现对变化 $Φ$ 和未知 $Ψ$ 的适应。请指正我如果描述上还有任何不准确的地方。

## 0.3 稀疏与变分

论文中提出的A-DLISTA和VLISTA模型主要通过以下方式实现了稀疏表示和变分操作:

1. 稀疏表示

- A-DLISTA的soft-threshold层进行基于学习字典Ψ的稀疏编码,类似于ISTA迭代,利用软阈值运算实现稀疏化。

- VLISTA中,字典Ψ通过先验和后验的采样传递到似然函数中,似然函数通过A-DLISTA获得信号的稀疏表示。

2. 变分操作

- VLISTA的先验网络和后验网络实现了字典上变分分布的建模。

- 先验网络学习字典的变分先验分布参数。

- 后验网络根据观测数据学习变分后验分布参数。

- 通过采样和迭代更新后验分布以实现变分推断。

- 训练目标是 Evidence Lower Bound (ELBO),包含先验和似然的贡献。

3. 两者结合

- VLISTA通过变分贝叶斯框架,实现了稀疏编码和字典学习的联合。

- 变分字典分布的采样为稀疏表示提供了不同条件下的Ψ。

- 稀疏表示的反馈 wieder调整字典的变分分布。

- 最终实现了信号稀疏表示和字典分布估计的统一。

综上,A-DLISTA通过soft-threshold层明确进行稀疏化,VLISTA进一步通过变分操作实现了稀疏编码和字典学习的贝叶斯统一框架。

# 1 Introduction

- 压缩感知利用先验稀疏性和凸优化技术来解决欠定系统。但是字典通常未知,测量矩阵也可能不同。这需要同时解决字典学习和稀疏表示学习问题。

- 作者提出了两种模型来解决这一问题:A-DLISTA和VLISTA。

- A-DLISTA是一种可以处理不同测量矩阵并适应当前数据实例的LISTA变体。

- VLISTA可以学习字典上的分布,被视为Bayesian LISTA模型,使用A-DLISTA作为似然模型。

- VLISTA根据优化动态调整字典,可以看作是分层表示学习方法。

- 字典分布可以用于检测异常样本(OOD)。这对基于压缩感知的方法很有用,因为它们通常无法检测OOD。

- 作者通过理论和实验支持了这两种模型,显示了它们学习到的不确定性是合理准确的。

总体来说,1 Introduction部分阐明了论文要解决的问题和所提出的方法,概括介绍了模型的创新之处。

## $y = ΦΨz$

论文中将图像的线性逆问题表达为y = ΦΨz而不是y = Φx的一个关键原因是:

这里假设原始信号x可以在某字典Ψ下得到稀疏表示z。

也就是说,存在一个稀疏系数向量z,使得x = Ψz。

将这个式子代入y = Φx,可以得到:

y = Φx = Φ(Ψz) = (ΦΨ)z

所以,从稀疏表示的角度出发,把逆问题表示成y = ΦΨz是很自然的。

之所以这么做,有以下几点考虑:

1. Ψ表示某一字典或变换,其目的是使信号在该字典下获得稀疏表示。

2. 压缩感知理论表明,在适当条件下,如果信号在某字典下稀疏,可以从少量线性测量中恢复信号。

3. 学习Ψ的分布是论文的一个关键目标。

4. 在算法迭代过程中,要同时优化稀疏编码z和字典Ψ。

将问题表示为y = ΦΨz,可以很自然地将字典Ψ纳入建模和算法中。

相比之下,仅用y = Φx则隐藏了信号的稀疏结构信息和字典的作用。

综上,论文采用y = ΦΨz这种表达方式,可以更好地体现稀疏表示思想,并融入字典学习。

## 补充知识点

### 1. Bayesian framework

贝叶斯框架(Bayesian framework)是一种基于贝叶斯论的概率建模和推理框架。其主要思想是:

1. 将所有参数和未知量建模为随机变量。

2. 利用贝叶斯定理,通过先验分布和数据(似然函数)推导出后验分布。

3. 使得参数空间中所有可能的参数值都有一个概率,从而进行不确定性建模。

4. 根据后验分布进行统计推断,如计算期望、 credibility intervals、Highest Posterior Density(HPD)区间等。

关键步骤包括:

(1) 定义参数的先验分布 p(θ),表示参数的不确定性。

(2) 得到数据后,根据数据计算似然函数 p(x|θ)。

(3) 使用贝叶斯定理得到后验分布 p(θ|x) ∝ p(x|θ)p(θ)。

(4) 基于后验分布进行统计推断。

贝叶斯框架区别于频率学派统计学,不仅提供点估计,而且给出整个参数后验分布。它通过合理建模先验信息,进行**不确定性量化**,是一种规范的处理不确定性的方法。

贝叶斯框架广泛应用于机器学习与人工智能等领域。该论文中使用贝叶斯框架进行字典学习,是一种典型的贝叶斯建模应用。

### 2. Bayesian approach

将压缩感知问题建模为一个贝叶斯框架,**对字典学习中的不确定性进行量化**。

具体来说,作者没有假设存在一个确定的 Ground Truth 字典,而是定义了字典的概率分布,并采用贝叶斯的思路来学习这个分布。

这种Bayesian approach的优点是:

1. 可以对字典进行不确定性量化,给出后验分布。

2. 不需要设计特定的先验分布来满足稀疏性约束。VLISTA通过阈值操作保留了稀疏性。

3. 字典后验分布可用于检测异常样本。

4. 根据优化动态适应和调整字典,实现分层表示学习。

5. 提供了一种概率方式来联合学习字典分布和重建算法。

综上所述,论文中提出的Bayesian approach指的是采用贝叶斯框架对字典学习问题建模,学习字典的概率分布,并基于此检测异常样本。这种方法很好地融合了贝叶斯与压缩感知,获得了不确定性量化及OOD检测的能力。

### 3. 测量矩阵

在压缩感知中,测量矩阵Φ表示线性测量过程。

具体来说,如果信号为向量x,测量结果为y,则二者之间的关系为:

y = Φx

其中Φ就是测量矩阵(measurement matrix)。

测量矩阵Φ决定了从信号x到测量y的线性变换或 projection。Φ每个列代表一个测量向量。

在该论文中,作者考虑了非静态测量场景,即对不同的数据样本x_i,Φ都不同,表示为Φ_i。

也就是每个样本x_i对应的测量结果y_i是通过一个样本特定的测量矩阵Φ_i得到的:

y_i = Φ_i x_i

这里Φ_i即表示针对第i个样本的测量矩阵。Φ_i对不同i是变化的,并且在过程中是未知的。

这种非静态的Φ_i设置打破了传统压缩感知假设测量矩阵Φ是固定已知的。因此对算法提出了更高的要求,需要适应不同的Φ_i。

综上所述,Φ_i表示对不同样本变化的测量矩阵,这种非静态设置在该论文中是一个重要的假设和设置。

### 4. non-static measurement scenario

测量矩阵在不同的数据样本中是变化的,而不是固定的。

也就是说,对于每个样本x_i,都有一个对应的随机测量矩阵Φ_i。然后根据Φ_i x_i得到观测y_i。

这与传统的压缩感知假设不同,传统方法假设对所有样本都存在一个固定的测量矩阵Φ。

非静态的测量矩阵意味着:

- 每个样本的观测条件都不同
- 不能假设所有样本都符合一个固定的测量矩阵

这给算法的设计带来了挑战。要对每个样本适配不同的Φ_i,而不是基于一个固定的Φ设计。

论文中后续的A-DLISTA和VLISTA模型考虑了这一非静态测量场景,可以适应每个样本的Φ_i,这是这两种模型的创新之处之一。

总之,non-static measurement scenario表示测量矩阵Between不同的数据样本是变化的,这BREAKS传统压缩感知固定Φ的假设,因此对算法设计提出新的要求。

# 2 Related Works

## Bayesian Compressed Sensing (BCS) and Dictionary learning

这一部分介绍了贝叶斯压缩感知和字典学习。贝叶斯字典学习是一种非参数贝叶斯方法，旨在对字典进行学习，而不是假设字典是固定的。该方法通过定义分布来量化恢复的不确定性，与传统的压缩感知方法有所不同。

## LISTA models

这一部分讨论了学习迭代软阈值算法（LISTA）及其变体。LISTA是一种通过将迭代算法展开为神经网络的层来学习其参数的方法。相关的研究工作提出了各种改进LISTA的指南，例如收敛性、参数效率、步长和阈值自适应等方面的改进。然而，这些方法通常假设稀疏化字典和感知矩阵是固定和已知的。

## Recurrent Variational models

这一部分介绍了循环变分模型。循环变分模型是一种学习数据样本之间时序相关性的生成模型。该模型通过在数据样本之间引入时间相关性来建模潜在变量，并利用变分推断方法进行训练。在本文中，当涉及到数据样本特定的字典时，提出的模型类似于循环变分自编码器的扩展。

## Bayesian Deep Learning

这一部分讨论了贝叶斯深度学习。在VLISTA中，如果使用全局字典，该模型本质上成为一种变分贝叶斯递归神经网络。变分贝叶斯神经网络是一种基于变分推断的生成模型，可以学习数据的生成过程和不确定性。在该工作中，模型考虑了每一步的先验分布和变分后验分布与先前步骤的条件关系。

# 3 Variational Learning ISTA

## 3.1 Linear inverse problems

这部分定义了压缩感知基本问题的数学表达式:

y = Φx 

其中:

- y 是M×1的测量结果向量
- x 是N×1的原始信号向量 
- Φ 是M×N的测量矩阵

## 3.2 LISTA

这部分详细介绍了LISTA模型:

- 最早在Gregor & LeCun (2010)论文中提出
- 将ISTA迭代方法展开成神经网络层,每层对应一次迭代
- ISTA迭代公式:
    $z_{t} = η_{θ_{t}}(z_{t-1} + γ_{t}(ΦΨ)^{T}(y - ΦΨz_{t-1}))$
- LISTA在每层学习权重矩阵 $W^t$ , $V^t$ 以逼近ISTA
- 做出两个关键假设:
  - **测量矩阵 $Φ$ 对所有的样本是固定的**
  - **字典 $Ψ$ 是预先给定的,并非学习**
- 这两个假设限制了LISTA的适用范围

## 3.3 Augmented Dictionary Learning ISTA

这一部分提出了A-DLISTA模型,目的是解决LISTA的两个限制:

1. LISTA假设字典 $Ψ$ 是预先给定和固定的
2. LISTA假设测量矩阵 $Φ$ 对所有样本都是固定的

A-DLISTA的创新在于提出了两个模块来解决上述问题:

1. Soft-threshold层

- 这个层类似ISTA迭代,进行稀疏编码
- 但是与LISTA不同的是,这里学习字典 $Ψ^t$,而不是预先给定 $Ψ$
- 每一层有自己的可学习字典 $Ψ^t$

2. Augmentation层 

- 这个层的输入是 $Φ^i$ 和 $Ψ^t$
- 输出是步长 $γ^t$ 和阈值 $θ^t$
- 通过神经网络学习如何从 $Φ^i$ 和 $Ψ^t$ 生成合适的 $γ^t$ 和 $θ^t$

这样,A-DLISTA就可以:

- **学习字典 $Ψ^t$,而不是预先给定**
- **根据每个样本的 $Φ^i$,生成自适应的 $γ^t$ 和 $θ^t$**

因此,相比于LISTA:

- 克服了字典 $Ψ$ 固定的限制
- 可以处理变化的 $Φ^i$
- 更加灵活适应不同样本

论文还给出理论结果,说明 $γ^t$ 和 $θ^t$ 的选取对收敛很关键。

综上,A-DLISTA通过soft-threshold层和augmentation层,扩展了LISTA的适用范围,使其可以学习字典 $Ψ^t$ 并处理varying $Φ^i$。这是该部分的主要创新与贡献。

## 3.4 VLISTA

这部分详细描述了VLISTA模型:

VLISTA是一个变分贝叶斯模型,用于同时解决字典学习和稀疏编码问题。其创新之处在于:

不再假设存在确定的固定字典 $Ψ$ ,而是学习字典上的后验分布;
该后验分布可以随优化迭代逐步调整,实现分层表示学习。

条件先验 $p_ξ(·)$,
变分后验 $q_ϕ(·)$, 
似然模型 $p_Θ(·)$.

### 3.4.1 Prior distribution over dictionaries

在VLISTA模型中,字典 $Ψ$ 不是预先确定的,而是需要学习的。为了表达字典的不确定性,作者使用了一个先验分布来建模字典 $Ψ$。

具体而言:

- $p(Ψ_t|Ψ_{t-1})$ 表示在时间步 $t$ 的字典 $Ψ_t$ 的先验分布。

- 该先验分布被设置为条件高斯分布(Gaussian distribution)。

- 分布的参数(均值和方差)由神经网络 $fξ$ 输出。

- 引入了条件依赖性: $Ψ_t$ 的先验分布依赖上一步 $Ψ{t-1}$。

也就是说,在每一迭代步 $t$ , $Ψ_t$ 都按照一个依赖前一步 $Ψ_{t-1}$ 的高斯分布进行采样。

这样做有以下好处:

- 引入字典之间的依赖关系,字典可以渐进地更新
- 通过神经网络学习先验分布的参数
- 每一步都进行采样,可以处理字典的不确定性

在第一步 $t=1$ ,由于没有 $Ψ_0$ ,默认设置为标准正态分布。

总之,先验分布 $p(Ψ_t|Ψ_{t-1})$ 表达了模型对字典 $Ψ$ 的不确定性,并将不同迭代步的 $Ψ$ 连接起来,这是VLISTA的一个关键创新。

### 3.4.2 Posterior distribution over dictionaries

在VLISTA模型中,还学习了一个后验分布 $q(Ψ_t|x_{t-1},y,Φ)$。

具体来说:

- $q(Ψ_t|x_{t-1},y,Φ)$ 表示在时间步 $t$ 对字典 $Ψ_t$ 的后验分布。

- 该后验分布也被设置为高斯分布。

- 但是与先验不同的是,后验分布的均值和方差由神经网络根据新增的观测来输出。

- 新的观测包括:上一步的重建信号 $x_{t-1}$ ,原始观测 $y$ ,以及测量矩阵 $Φ$。

- 根据新信息调整后验,进行贝叶斯推断。

这样做的优点是:

- 后验分布可以利用新的观测,迭代地调整字典
- 实现了字典学习和稀疏编码的联合优化
- 模型可以跟踪信号和字典的联合分布

在每一步,模型先采样出字典,然后基于该字典进行信号重建。再利用新的重建结果调整字典后验。如此迭代进行。

总之,后验分布与先验分布联合,使VLISTA实现了迭代式的字典学习和表示学习。

### 3.4.3 Likelihood model

在VLISTA的变分贝叶斯框架中,似然项对应信号重建模型。具体而言:

- 似然分布表示针对输入信号 $x$ ,给定字典 $Ψ_1:t$ 后的重建分布 $p(x_t|Ψ_1:t)$。

- 该重建分布被设置为高斯分布。

- 高斯分布的均值 $μ$ 就是A-DLISTA网络的输出。

- 即 $p(x_t|Ψ_1:t) ≈ N(μ = A-DLISTA(Ψ_1:t), δ^2)$

- 方差δ是一个预定的超参数。

也就是说,根据采样得到的字典Ψ^1:t,我们用A-DLISTA网络进行信号重建,将网络输出作为高斯似然分布的均值。

这样做的好处是:

- 利用了A-DLISTA网络进行端到端的信号重建
- 为信号建模提供不确定性
- 与字典的先验和后验配合,进行联合Bayes学习

训练时,对数似然项直接用重构误差来替代。

总之,似然项通过A-DLISTA获得信号的重建分布,与字典的先验和后验配合,形成了VLISTA的变分Bayes框架。

### 三部分之间的关系

在VLISTA的变分贝叶斯框架中,字典的先验分布、后验分布和似然分布之间的关系是:

1. 先验分布 $p(Ψ_t|Ψ_{t-1})$

- 定义字典的先验知识和不确定性
- 通过神经网络参数化
- 引入 $Ψ_t$ 与 $Ψ_{t-1}$ 的依赖性

2. 后验分布 $q(Ψ_t|x_{t-1},y,Φ)$

- 根据观测数据更新字典分布 
- 采样Dictionary用于信号重建
- 实现字典学习

3. 似然分布 $p(x_t|Ψ_1:t)$

- 基于采样字典,用A-DLISTA网络重建信号
- 输出作为高斯分布均值
- 进行信号建模

它们之间的关系是:

- 先验分布定义了字典的初始分布
- 根据先验和观测,后验分布迭代更新字典 
- 根据采样字典,似然分布重建信号
- 回馈重建结果再次调整字典后验

三者彼此配合,形成了一个联合学习字典和信号表示的变分贝叶斯框架。

总之,三者合作学习字典分布、进行信号重建和表示,实现了VLISTA的关键创新之处。

### 建模过程

论文中VLISTA模型进行变分推断时的建模过程如下:

1. 建立字典的变分先验分布p(Ψ):

- 将字典Ψ作为随机变量
- 假设Ψ每一步的先验分布为条件高斯分布
- 先验分布的参数由神经网络表示

2. 建立字典的变分后验分布q(Ψ):

- Ψ的后验也采用高斯分布表示
- 后验分布依赖于观测和上一步的重建结果
- 后验的参数也由神经网络输出

3. 建立观测的似然函数p(x|Ψ):

- 似然函数采用A-DLISTA网络输出的高斯
- 即根据采样字典Ψ重建信号

4. 讲先验、后验和似然结合起来,进行变分Bayes学习:

- 根据先验采样字典,计算似然获取重建信号
- 用重建信号更新字典后验分布
- 反复迭代进行变分推断

5. 训练目标是ELBO,包含先验、似然和KL散度项。

综上,论文通过高斯变分分布建模字典的不确定性,并与重建网络联系起来,进行变分贝叶斯学习。

### 训练目标

在变分Bayes学习中,训练目标通常是Evidence Lower Bound(ELBO)。

对于VLISTA模型,ELBO包含以下三项:

1. 似然项 ∑_{t=1}^T E_{Ψ∼q} [log p(x^t|Ψ)]

- 这项表示根据采样出的字典Ψ,计算重建信号x^t的对数似然。

- 反映了根据当前字典的重建效果。

- 最大化这一项可以提高重建性能。

2. KL散度项 ∑_{t=2}^T E_{Ψ∼q} [KL(q(Ψ^t) || p(Ψ^t|Ψ^t-1))]

- 这项表示字典后验分布与先验分布的KL散度。

- 最小化KL散度可以正则化后验分布,防止过于复杂。

- 因为我们希望后验接近先验。

3. 先验项 - KL(q(Ψ^1) || p(Ψ^1))

- 对第一个字典的KL散度,类似作用,防止过复杂的先验。

综合以上三项作为目标具有以下好处:

- 最大化似然项,以获得良好的重建效果
- 最小化KL散度项,防止过复杂的后验和先验
- 平衡重建效果和正则化效果

因此,ELBO是一个理想的训练目标,整合了重建性能和Bayes正则化。

## 补充知识点

### 为什么使用高斯分布

论文中在VLISTA模型进行变分推断时,采用高斯分布来建模字典的先验分布和后验分布,主要出于以下考虑:

1. 高斯分布是最简单和最常用的连续概率分布之一,具有很好的数学性质。

2. 高斯分布由均值和方差两个参数完全确定,这很适合用神经网络进行参数化。

3. 高斯分布是共轭先验,可以简化运算,方便进行 Bayesian 更新。

4. 高斯分布可以很好地表示 参数或变量的不确定性。

5. 高斯分布支持重参数化技巧,可以方便地改变参数空间。

6. 高斯分布广泛应用在变分推断中,理论和实践基础都很成熟。

7. 使用高斯分布进行建模比较简单直观,也比较容易实现。

8. 对于字典这样的连续变量,高斯分布是一个合理的选择。

总之,高斯分布既有很好的数学属性,也非常适合用来表示连续参数的不确定性。在变分推断中使用高斯分布可以简化计算,也是一种简单有效的选择。这些优点使其成为论文中表达字典不确定性的第一选择。

### 为什么希望后验分布q(Ψ)接近先验分布p(Ψ)?

- 因为先验分布p(Ψ)表示对Ψ的先验假设,包含了我们的Existing知识。

- 如果后验分布偏离先验太远,那表示学习的结果不符合我们现有的知识。

- 我们希望学习结合先验知识和新数据,不能脱离先验太远。

- 所以需要后验分布保持在一个合理的范围内,不能过于复杂。

### Bayes正则化

- Bayes正则化指的是通过先验分布来控制或者约束后验分布的复杂度。

- 采用KL散度度量后验和先验的接近程度,来实现正则化。

- 最小化后验和先验的KL散度,可以防止后验过于复杂。

- 这与其他正则化技术的目标类似,防止过拟合。

- Bayes正则化从贝叶斯的角度利用了先验知识。

综上,后验接近先验可以保证学习到的结果理性;Bayes正则化则提供了一种基于先验知识的正则手段,防止过拟合。这就是变分Bayes学习中这一项的作用。

# 4 Experimental Results

论文4 Experimental Results部分进行了以下实验:

1. 在MNIST和CIFAR10数据集上进行图像重建实验,比较不同模型在不同测量数下的SSIM。

2. 在合成数据集上测试信号重建误差。

3. 比较VLISTA和贝叶斯压缩感知在检测异常样本上的效果。

主要实验结果如下:

- A-DLISTA优于ISTA、LISTA等非贝叶斯模型,证明了其适应变化测量矩阵Φ的优势。

- VLISTA优于贝叶斯压缩感知,但不如A-DLISTA。

- VLISTA可以成功检测出异常分布样本,而其他模型无法做到。

- 在测量数较少时,作者模型的优势更加显著。

这些实验结果说明:

- A-DLISTA适应非静态Φ和未知字典Ψ的能力得到了验证。

- VLISTA通过学习字典分布获得了检测异常样本的能力。

- 综合结果验证了论文方法可以适应变化的Φ,学习Ψ,并检测异常样本。

- 在欠定条件下,论文方法优于已有压缩感知和贝叶斯模型。

## 4.1 MNIST & CIFAR10

这部分在MNIST和CIFAR10两个图像数据集上评估了不同模型的图像重建性能。主要实验设置和结果如下:

- 用不同模型(ISTA、LISTA、DLISTA、A-DLISTA、BCS、VLISTA)重建图像。

- 层数固定为3层,改变测量数measurement,比较SSIM。

- 对每个样本生成随机Gaussian测量矩阵 $Φ$。

- A-DLISTA优于所有非贝叶斯模型(ISTA、LISTA、DLISTA)。

- VLISTA优于BCS,但不如A-DLISTA。

- 尤其是测量数很少时(10-100),作者模型明显优势。

- 随着测量数增加,各模型性能趋于接近。

这说明了在非静态 $Φ$ 和字典未知场景下,A-DLISTA和VLISTA的效果优于传统压缩感知方法。它验证了模型的 adaptation 机制的有效性。

## 4.2 Synthetic Dataset

这部分的实验设置是:

- 随机生成测量矩阵 $Φ$ 和字典 $Ψ$。

- 根据 $Φ$ 和 $Ψ$ 随机生成稀疏信号。

- 学习各个模型,测试信号重建误差。

- 用重建误差的分位数来评价不同模型。

主要结果是:

- 在合成数据上,A-DLISTA仍然优于其他非贝叶斯模型如ISTA、LISTA等。

- 这进一步验证了A-DLISTA在测量矩阵变化且字典未知场景下的强大适应能力。

- VLISTA的性能也优于传统Bayes压缩感知方法。

- 当测量数很少时,作者模型的优势更加明显。

- 随着测量数增加,误差下降,但大体关系保持一致。

总之,这部分实验在合成数据上也验证了论文方法在非静态 $Φ$ 和 $Ψ$ 未知场景下的优势。

## 4.3 Out Of Distribution detection

这部分评估了VLISTA在检测异常分布样本(OOD)上的能力:

- 使用MNIST数据集,训练集只包含部分数字(0,3,7)。

- 其他数字的样本作为异常样本。

- 训练VLISTA后,用训练好的模型测试ID样本和OOD样本。

- 对每个样本多次重建,记录variance。

- 计算p值来判断是否Reject OOD样本。

主要结果是:

- VLISTA可以成功检测出OOD,p值较低,rejection power更强。

- 当加入噪声时,VLISTA的OOD检测仍比贝叶斯压缩感知更佳。

- 这验证了通过学习字典分布,VLISTA获得了检测OOD的能力。

- 而其他压缩感知方法如ISTA、LISTA等则不具有这一能力。

总之,这部分实验展示了变分Bayes方法学习字典分布的优势之一是支持OOD检测,这是其他方法所不具备的。

## 补充知识点

在论文的实验部分中提到的measurement指的是压缩感知中的线性测量。

具体来说,measurement对应的是矩阵 $Φ$ 对信号 $x$ 的线性投影运算:

$y = Φx$

这里:

- $x$ 是原始信号
- $Φ$ 是测量矩阵
- $y$ 是测量结果

每一列的 $Φ$ 表示一个测量向量。

measurement的个数即指 $Φ$ 的列数,也就是说对 $x$ 进行了多少次线性测量。

在论文实验中:

- measurement的个数是可控变量
- 通过改变measurement的个数,来评价不同模型在不同测量数下的重建效果

例如,当measurement只有10个时( $Φ$ 有10列),由于严重欠定,重建效果比较差。

随着measurement增加到100,300等,重建效果逐渐提升。

所以论文中通过比较不同measurement数下的性能,可以比较不同模型在复原度不同条件下的优劣势。

简而言之,measurement指的就是进行线性压缩感知时,矩阵 $Φ$ 的列数,也就是进行的测量个数。

# 5 Conclusion

本文提出了一种变分方法VLISTA,用于联合解决字典学习和稀疏表示恢复问题。传统压缩感知框架假设存在某真值字典用于重建信号,而LISTA类模型假设测量矩阵是固定的。本文放松了这两个假设。

首先,设计了一种软阈值算法A-DLISTA,可以处理不同的测量矩阵并适应每个数据实例。我们从理论上证明了使用增强网络来适应每个层的阈值和步长的有效性。其次,通过为字典引入一个概率分布,放松了真值字典存在的假设。基于该假设,构建了变分框架VLISTA来解决压缩感知问题。

我们在多个数据集上报告了两种模型A-DLISTA和VLISTA的性能,包括非贝叶斯方法和贝叶斯方法。实验证明A-DLISTA的适应性提升了相比ISTA和LISTA的性能。尽管VLISTA的重建性能不如A-DLISTA,但其变分框架可以评估信号的不确定性,用于检测异常样本,而其他LISTA类模型无法做到这一点。此外,与其他贝叶斯压缩感知方法不同,VLISTA不需要设计特定先验来保持稀疏性,其聚合操作作用于字典而不是信号本身。

综上,本文通过变分贝叶斯框架成功解决了字典学习和稀疏表示问题,并展示了适应性和检测异常样本的能力。