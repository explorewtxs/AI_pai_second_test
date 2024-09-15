# 概述
* 本节讨论的模型：读取一段文本，并且预测下一个单词。(Transformer)
# Attention机制的作用
## An Example
考虑以下两个句子
* One mole of carbon dioxide
* American shrew mole

如果只经过了Embedding步骤，那么得到的单词“mole”对应的向量是一样的。然而在不同的语境中，mole显然有不同的含义。

## Attention模块具体作用
注意力模块可以做到
* 精确一个词的含义
* 将一个嵌入向量中的信息传递到另一个嵌入向量中。

## Another Example
有一段悬疑小说(最后一句是"The murderer was)预测was后面的一个单词使用的是矩阵中的最后一个列向量。经过embedding步骤之后，该列向量仅仅表示was这个单词，但是经过若干Attention等中间模块之后，其嵌入了完整的上下文信息。

# 注意力模式
## An example
* a fluffy bule creature roamed the verdant forest
现在只考虑让形容词调整对应名词的含义。
* 一个注意力模块是由许多不同的分支并行运行组成的。这个example讨论的是其中一个分支
## 初始嵌入
* 每个词的初始嵌入式一个高维向量。如下图所示。
![](初始embedding.png#pic_center)
* 其中每一个列向量表示对应的token是什么并包含了该token的position信息。
## 使用形容词调整对应名词的含义
### 计算形容词和名词的相关性
#### Query向量
一个名词需要知道其前面是否有形容词修饰。
* Query向量维度：例如128维(much smaller than embedding vector)。
* Query向量作用(概念型理解)：例如，creature生成的Query向量Q<sub>4</sub>用于提问前面是否有形容词。
* Query向量获得方法：用矩阵W<sub>Q</sub>和Embedding向量的乘积得到。如以下两图所示。
![](Q向量计算.png#pic_center)
![](从E向量到Q向量.png#pic_center)
* Query向量作用(理想的数学角度)：将名词从高维的Embedding space嵌入到较低维的Query/Key space上。如下图所示。
![](Q向量作用.png#pic_center)
#### Key向量
* Key向量计算方式：将Embedding向量和W<sub>k</sub>矩阵进行乘积得到。如下图所示。
![](K向量获得.png#pic_center)
* key向量作用(从概念上理解)：Key向量可以看作是查询回答者。和查询矩阵W<sub>Q</sub>一样，W<sub>k</sub>充满了可调整的参数。
![](Key向量作用.png#pic_center)

#### Attention Pattern
* 使用点积来衡量Q向量和K向量的匹配程度。如下图所示。匹配程度越高点积越大。
![](dot_K_Q.png#pic_center)
* 在这一节使用的例子总，理想情况是token fluffy和token blue产生的两个K向量和token creature产生的Q向量点积后的结果较大的正值。
* 点积后会得到一个矩阵，对这个矩阵中的每一列做softmax得到概率。
* 在Transfomer原始论文中，Q与K向量进行点积之后，还除以了维度d开根号之后的值。

#### 其他知识点

##### masking
* 在设计Attention pattern时有一个基本原则，不允许后续出现的token影响先前出现的token，因此需要将下图中红框中的值变为0。
![](set_to_zero.png#pic_center)
* 常用做法是在softmax之前将这些值变为-∞
##### 上下文大小
* Attention pattern的大小和成平方关系。

### 使用形容词调整名词含义
#### 概述
在经过Embedding之后，token creature得到的是一个12288维度的向量。前面使用W<sub>Q</sub>矩阵得到的Q向量和使用W<sub>K</sub>矩阵得到的K向量用于计算所有token之间的相关性关系，得到某个形容词是否修饰了某个名词(例如 fluffy修饰了creature)。然而前面这一步还并没有对token creature对应的12288维度的向量进行调整。
#### Value matrix W<sub>V</sub>
* W<sub>V</sub>乘以**E**<sub>fluffy</sub>得到向量**V**<sub>fluffy</sub>。以同样的方法得到**V**<sub>a</sub>以及**V**<sub>blue</sub>等。接着**E**<sub>creature</sub> = **E**<sub>creature</sub> + **V**<sub>fluffy</sub> * (**K**<sub>fluffy</sub> dot )
* W<sub>V</sub>乘以**E**<sub>fluffy</sub>得到向量**V**<sub>fluffy</sub>。以同样的方法得到**V**<sub>a</sub>以及**V**<sub>blue</sub>等。接着**E**<sub>creature</sub> = **E**<sub>creature</sub> + **V**<sub>a</sub> * (**k**<sub>a</sub> dot **Q**<sub>creature</sub>)+**V**<sub>blue</sub> * (**k**<sub>blue</sub> dot **Q**<sub>creature</sub>)+**V**<sub>fluffy</sub> * (**k**<sub>fluffy</sub> dot **Q**<sub>creature</sub>)...


## 参数
前面提到了三个矩阵，分别为W<sub>Q</sub>,W<sub>K</sub>,W<sub>V</sub>
### 参数个数in GPT-3
* W<sub>Q</sub>: 12288 * 128 = 1572864
* W<sub>K</sub>: 12288 * 128 = 1572864
* W<sub>V</sub> = 12288 * 12288 = 150994994
* 为了更高效，不会直接使用如此庞大的W<sub>V</sub>矩阵，而是将这个矩阵拆分成两个较小的矩阵W<sub>V</sub> = 值升维矩阵 * 值降维矩阵，值升维矩阵大小:12288 * 128，值降维矩阵大小为:128 * 12288

## cross-attention
前面所有的部分都属于 self-attention(自我注意力)
cross-attention和self-attention非常相似。不同之处在于cross-attention会处理两种不同类型的数据，如语音输入和转录。使用cross-attention时，**K**向量**Q**向量来自于不同的language，点积衡量的是不同语言中两个词的对应程度。这一过程没有masking

## 多头注意力
通常self-attention会并行进行许多次，GPT-3中会进行96次。有96个W<sub>K</sub>,W<sub>Q</sub>,W<sub>V</sub>