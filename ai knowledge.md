## 1. 注意力监测机制 

  定义：从海量输入中筛选出关键信息，并对不同信息分配不同的处理权重，以解决信息过载问题。
  
  一、 深度学习（Transformer/LLM）中的注意力机制：数学映射与权重分配，底层逻辑是高维向量空间的映射与相似度计算。

早期的输出根本不连贯，：
<img width="1673" height="740" alt="image" src="https://github.com/user-attachments/assets/60e9cd46-2ad3-4c3c-b3c9-92346f4d3147" />

  1. 核心映射：Q、K、V 三元组你可以把注意力机制想象成一个数据库检索系统：
  <img width="1518" height="623" alt="image" src="https://github.com/user-attachments/assets/08ade64f-e475-4c03-80ee-eac12ee67e31" />
<img width="1529" height="822" alt="image" src="https://github.com/user-attachments/assets/b09830be-e690-48e1-8ef5-220f357b62f2" />

  - Query (Q, 查询)：当前模型正在关注的词，它想问：“我需要什么信息？”
  - Key (K, 键)：所有其他词的标签，它们在说：“我这里有什么信息？”
  - Value (V, 值)：所有其他词的具体内容。
    交叉注意力：Q 挨个检查 K 和哪个词最相关，最终提取最有用的 V 生成
    
    自注意力：一个句子里被拆分不同的词，分别是 QKV，互相匹配
    
    两者是大语言模型的核心
    
2. 底层运算逻辑注意力机制通过三个步骤实现“监测”与“聚焦”：
<img width="1513" height="847" alt="image" src="https://github.com/user-attachments/assets/f74d01d8-b95e-4b99-b4ea-ca599d53c519" />
最大池化：图片里的三维被压缩成一个扁平的地图，平均池化是每个坐标点对应的通道的平均值，两张底稿融合交给卷积层，最终合成一张图，覆盖在原图上，是空间注意力机制。

- 相似度匹配 (Dot Product)：将 $Q$ 与所有的 $K$ 做点积运算，计算出当前词与其他词的关联强度（越相似，分数越高）。
- 归一化 (Softmax)：将分数转化为概率分布（权重），确保所有权重之和为 1。这意味着 AI 决定了把“注意力的额度”按多少百分比分给序列中的每一个词。
- 加权求和 (Weighted Sum)：将权重乘以对应的 $V$（值），最后汇总。这便得到了一个融合了上下文关联信息的“新表示”。权重就是谁最重要
- 底层本质：这是一种动态的权重分配算法，通过数学手段实现了对信息关联性的“监测”与“捕获”。


## 2.函数拟合
函数拟合（Function Fitting） 就是在已知的一堆散乱的数据点中，寻找一条“最匹配”的数学曲线，用来描述这些数据背后的规律。
拟合的本质是寻找一个函数 $f(x)$，让它生成的预测值与真实数据点的误差最小。

1. 拟合的底层：最经典的逻辑是 “最小二乘法”（Least Squares）：

- 计算每一个数据点到曲线的垂直距离（即误差）。
- 将所有这些误差进行平方（为了消除正负抵消）。
- 把所有平方后的误差求和。寻找参数，使得这个“总误差平方和”达到最小。
<img width="1543" height="881" alt="image" src="https://github.com/user-attachments/assets/4c6eb187-aa12-4138-b8a9-06aa9a016da4" />
<img width="1619" height="922" alt="image" src="https://github.com/user-attachments/assets/132dce1e-e206-4d9a-8950-553127c483f0" />
<img width="1492" height="897" alt="image" src="https://github.com/user-attachments/assets/22624f6f-6dc6-4e49-91ba-f17f91d1e702" />

 例如，原先的一个直线无法拟合成一个曲线，但添加了 ReLU 函数，那么直线就可以产生了拐点，b 是负的 a,两者相加就出现了图 2 的图形，最终搞了四个 ReLU 公式，图形如图 3所示
 
<img width="1315" height="826" alt="image" src="https://github.com/user-attachments/assets/31ef00b9-1edb-4885-9b7c-9a50b13e38e6" />

 不断调整参数，最终会有一个 ReLU 函数无限逼近目标函数。

 <img width="1369" height="771" alt="image" src="https://github.com/user-attachments/assets/a9efd9a6-5b8d-49bb-90c0-abf3460a22d6" />

二维到三维变换出现了曲面

2. 拟合中的陷阱：过拟合与欠拟合这是拟合中最关键的两个概念：

- 欠拟合 (Underfitting)：模型太简单了，强行用直线去拟合复杂的曲线。它既不能准确描述现有数据，也无法预测未来。
- 过拟合 (Overfitting)：模型太复杂，强行穿过了每一个数据点。它看起来在训练数据上表现完美，但因为把“噪音”也当成了规律，导致它在预测新数据时表现得很差。
