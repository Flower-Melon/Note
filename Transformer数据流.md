# `transformer`中的数据流

## 0 预备

虽然早就对`transformer`的结构及其为什么有效比较了解了，可是每当涉及到真正训练和推理时的数据流时仍然不是很清楚，今天直接写一个笔记捋清一下整个过程

* 缩放点积注意力

$$softmax\left( {\frac{{{\bf{Q}} \cdot {{\bf{K}}^T}}}{{\sqrt d }}} \right) \cdot {\bf{V}}$$

其中 ${\bf{Q}}$ 的形状为`[num_q, d]`, ${\bf{K}}$ 的形状为`[num_k, d]`（缩放点积注意力要求 ${\bf{Q}}$ 和 ${\bf{K}}$ 的维度必须相等）， ${\bf{V}}$ 的形状为`[num_k, value_dim]`（键值对数量相等）

${{\bf{Q}} \cdot {{\bf{K}}^T}}$ 得到了一个系数矩阵`[num_q, num_k]`，其第 $i$ 行第 $j$ 列代表第 $i$ 个`query`和第 $j$ 个`key`的关联程度

最后与 ${\bf{V}}$ 进行矩阵乘法得到`[num_q, value_dim]`得到每个`query`的分数

## 1 训练

训练时写的稍微清楚一点，以`decoder-only`结构为例

* 词元化处理

输入一个句子为`[batch_size, length]`，每个`batch`有`length`个`token`,长度不够的以`<pad>`填充

先将其对应到词表`vocab`上，这个词表非常大比如英语的词表大小几十万，获得`[batch_size, length, vocab_size]`

每个单元都是一个长度很大（十几万）的独热向量，不利于训练，经过嵌入层（是个网络，其参数需要训练）`(vocab_size, embed_size)`将其转化成`[batch_size, length, embed_size]`，`embed_size`可能就只有大几百了

位置编码等其他技术不做赘述

* 注意力映射

这里不对多头注意力做赘述

使用自注意力机制，这里使用三个网络 ${W_q}$ ， ${W_k}$ 和 ${W_v}$ （形状为`[embed_size, d]`，一般来说`embed_size`和`d`相等）将处理好的向量映射到三个空间 ${\bf{Q}}$ ， ${\bf{K}}$ 和 ${\bf{V}}$ 上，形状变为`[batch_size, length, d]`

在训练一开始初始化的 ${\bf{Q}}$ ， ${\bf{K}}$ 和 ${\bf{V}}$ 完全相等，等同于自注意力，经过训练之后才不同

* 掩码`mask`

${{\bf{Q}} \cdot {{\bf{K}}^T}}$ 得到了系数矩阵`[batch_size, length, length]`，考虑实际上输入的句子，靠前位置的`token`实际上不应该能注意到靠后位置的`token`，因为还没生成到，不可能预知未来，所以这时候需要进行掩码操作，对系数矩阵进行处理，构建一个掩码矩阵:

$\begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ \cdots & \cdots & \cdots & \cdots \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$

大小也为`[batch_size, length, length]`，与系数矩阵相加后可以避免前面的`token`可以注意到后面的`token`

* 损失函数

最后与 ${\bf{V}}$ 进行矩阵乘法得到`[batch_size, length, d]`

后面的`FFN`和层规范化不做赘述

最后经过一个全连接层（`[d, vocab_size]`）和`softmax`输出`[batch_size, length, vocab_size]`，存储了每个`token`在产出时的概率，这个概率与原始序列做交叉熵损失即为模型的损失函数，即可更新参数

* 并行的特性

无论是`decoder-only`架构还是`encoder-decoder`架构在训练时，由于`transformer`的注意力机制的强大特性，模型可以在训练时并行处理整个序列，而不用像`RNN`一个个处理`token`，效率非常高，这也是为什么其强大的原因之一（另一个原因从知乎帖子看来的：[注意力机制处理长依赖序列的优越性](https://www.zhihu.com/question/580810624/answer/2979260071)）

## 2 推理

推理时和训练时就不同了，模型不可能一下子输出一个自然语言的序列，只能不断输出一个个`token`，输出的单个`token`连接到`prompt`上形成新的输入，模型不断地进行自回归，直到输出截止的标志`<eos>`,模型停止输出，过程上来说这时候和以`RNN`为基础的模型区别不大