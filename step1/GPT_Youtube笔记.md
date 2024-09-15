* GPT is short for Generative Pre-trained Transformer
  
## Transformer作用
* voice-to-text
* text-to-voice
* text-to-image

## Transformer内部运行机制
* step1提供一句话，首先划分成若干个token
* step2每一个token都会被映射成一个向量，token的含义越相近，向量也就越加靠近。
* step3经过Attention block，使得不同向量之间能够交流
* step4 经过Mutilayer Perceptron，这一步不同向量之间没有交流，只是进行并行的处理。
* 重复Attention block以及Mutilayer Perceptron 
## GPT-3有1750亿个参数
* 基本为矩阵和向量的乘积。
# 文本处理
有这样一段文字“The goal of our model is to predict the next”现在的任务是text-generation
## 文本处理第一步——Word embedding
* 首先将这段文字划分成若干个token，然后转换成向量。
* token不一定完整地对应于单词，例如"(known fancifully as tokenization)"可能化成的tokens有"(","known"," fanc","ifully"," as"," token","ization",")"。但是这个视频为了使读者理解更加容易，假设token就是完整地对应一个单词。
* 模型拥有一个预设的词汇库，包含了所有可能的单词。Embedding matrix中每一列代表一个单词的向量。
* 利用Embedding matrix将token转换成向量。
* Embedding matrix中的每一列可以看作该token在高维空间的中的向量。Embedding 不是任意的，每个空间的方向都有特定语义，相近的词向量会更加相似。E(aunt)-E(uncle)约等于E(woman)-E(man)。E(woman)-E(man)这个方向包含了gender信息。(`待实验`)E(Hitler) + E(Italy) -E(Germany)约等于E(Mussolini)
* 向量点积：用于检测两个向量方向的对齐程度。
* embedding这一步使用的参数个数为：d_embed(embed这一步的维度)*n_vocab(token个数) = 12288 * 50257 = 617558016。
## 第二步——Embedding beyond words
* 仅仅使用第一步存在两个问题：第一个是没有体现token位置的作用，同一个token位于句子的首位和末尾作用一定是不一样的；第二个是没有包含语义信息，同一个token在不同语境中含义不一样。因此通过第一步的处理得到的矩阵中的每一列只包含了单独的token的信息。
* 这一步的作用是使得第一步得到的矩阵中的每一个向量都包含周边信息(例如token的位置以及语义)
* 利用多层的Attention + Mutilayer Perceptron实现
* GPT-3训练过程中Context Size大小为2048。
## 最终步骤——Uembedding
* 最终目标是产生一个概率分布，预测下一个可能出现的token。
* 这一步骤分为两步：第一步使用一个矩阵，将最后一个向量映射成50k个值，每个值对应词汇表中得一个token；然后使用softmax函数将这些值进行映射
* 上面步骤使用到的矩阵记为W<sub>U</sub>，大小为50257(总token数)*12288 = 617558016
## Softmax函数
* Softmax函数使用的目的：将一连串的数字变成概率。每个值在0和1之间，并且和为1。
* 进行如下处理
$$
\frac{e^{x_i}}{\sum_{n=0}^{N-1}e^{x_n}}
$$
* 还可以加入temperature——T，T较大时，较小的值可以获得更多的权重；如果T较小，那么较大的值会更加dominate
$$
\frac{e^{x_i/T}}{\sum_{n=0}^{N-1}e^{x_n/T}}
$$