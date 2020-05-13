# nlp_2020_exp1_classification

### Dataset

&emsp;&emsp;本次文本分类实验为多分类实验, 数据集中文本类别为 5 类: `news_culture, news_car, news_edu, news_house, news_agriculture`.  
&emsp;&emsp;数据集中每一行从左至右的字段为： `example_id, category_code(non-sense), category, example` , 可参考下面的例子.

```sh
# Example with fields separated by '\t'
6523865677881672199	101	news_culture	黄氏祖训、家训——黄姓人家可以鉴读一下
```

### Pkgs

&emsp;&emsp;有些 package 并没有在依赖文件 `requirements_dev.txt` 中给出, 要使用 baseline 的话还需要安装下面的 package

```sh
pytorch==1.4.0
cudatoolkit==9.2
tensorboard==2.2.1
scikit-learn==0.22
jieba==0.42.1
```

### Visualize baseline

&emsp;&emsp;baseline 的训练过程可以通过下面的命令可视化.

```sh
tensorboard --logdir=./runs
```

![baseline](resources/baseline.jpg)

### Reference

&emsp;&emsp;以下为本次实验需要用到的数据集(数据集已经划分为训练集/验证集/测试集), 还有 baseline 使用的预训练词向量, 预训练词向量为在搜狗新闻上使用 SGNS(skip-gram with negative sampling)训练得到的 300 维词向量.

- Data: [link](https://pan.baidu.com/s/1TprekQac-yzNHMsREWZe9g), verification Code: uhxt  
- Pretrained-embedding: [link](https://pan.baidu.com/s/1svFOwFBKnnlsqrF1t99Lnw)  
- Reference: https://github.com/Embedding/Chinese-Word-Vectors

### Assignment

#### content

&emsp;&emsp;本次实验需要利用给出的数据集, 最终提交的形式为压缩包, 压缩包中应该包含 **实验报告(pdf)** 和 **源代码**. 实验报告中至少应有的内容: 预处理过程, 模型结构, 超参数配置, 评估方法, 测试集上的最终结果, tensorboard 可视化训练结果.  



## Mywork

#### 1. 预处理过程

本次文本分类实验为多分类实验, 数据集中文本类别为 5 类: 

news_culture, news_car, news_edu, news_house, news_agriculture.

数据集中每一行从左至右的字段为：

 example_id, category_code(non-sense), category, example。

> \# Example with fields separated by '\t'
>
> 6523865677881672199	101	news_culture	黄氏祖训、家训——黄姓人家可以鉴读一下

预处理部分采用baseline中的方法：

（1）定义样本的处理操作：torchtext.data.Field

（2）构建Dataset，TabularDataset：Defines a Dataset of columns stored in CSV, TSV, or JSON format.对于csv/tsv类型的文件，TabularDataset很容易进行处理，故我们选它来生成Dataset

（3）创建词汇表，用来将 string token 转成 index：field.build_vocab()

（4）构造迭代器：采用BucketIterator迭代器生成数据。相比于标准迭代器，会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。

```
bucket_iterator = BucketIterator(
  train_dataset,
  batch_size=args.train_batch_size,
  sort_within_batch=True,
  shuffle=True,
  sort_key=lambda x: len(x.news),
  device=args.device,
)
```

#### 2.模型结构

本次实验采用TextCNN、TextRNN(单、双向)、TextRCNN、TextRNN+Attertion共4个模型进行对比实验。（参考来源https://zhuanlan.zhihu.com/p/73176084）

（1）TextCNN

![img](README.assets/wps1.png)

```python
def forward(self, x, x_len):
  x = self.embedding(x)
  x = torch.transpose(x, 1, 0)  # 将 0轴 和 1轴 交换
  x = x.unsqueeze(1)  # (N,Ci,W,D)
  x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
  x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
  x = torch.cat(x, 1)  # (N,Knum*len(Ks))
  x = self.dropout(x)
  logit = self.fc(x)
  return logit
```

（2）TextRNN

![img](README.assets/wps2.png)

```python
def forward(self, x, x_len):
  x = self.embedding(x)
  \# Pad each sentences for a batch,
  \# the final x with shape (seq_len, batch_size, embed_size)
  x = pack_padded_sequence(x, x_len)
  \# h_n: (num_layers * num_directions, batch_size, hidden_size)
  \# NOTE: take the last hidden state of encoder as in seq2seq architecture.
  '''
  output保存了最后一层，每个time step的输出h，如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)。
  h_n保存了每一层，最后一个time step的输出h，如果是双向LSTM，单独保存前向和后向的最后一个time step的输出h。
  c_n与h_n一致，只是它保存的是cell的值。
  '''
  hidden_states, (h_n, c_c) = self.lstm(x)
  h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous() # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
  \# h_n:(batch_size, hidden_size * num_layers * num_directions)
  h_n = h_n.view(h_n.shape[0], -1)
  loggits = self.linear(h_n)
  return loggits
```

（3）TextRCNN

![img](README.assets/wps3.png)

```python
def forward(self, x, x_len):
  x = self.embedding(x)
  x = torch.transpose(x, 1, 0)  # 将 0轴 和 1轴 交换，[batch_size, seq_len, embed_size]
  out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * 2]
  out = torch.cat((x, out), 2)  # [batch_size, seq_len, hidden_size * 2 + embed_size]
  out = F.relu(out)
  out = out.permute(0, 2, 1)  # [batch_size, hidden_size * 2 + embed_size , seq_len]

  pad_size = x_len[0].item()

  maxPool = nn.MaxPool1d(pad_size)
  out = maxPool(out).squeeze()  # [batch_size, hidden_size * 2 + embed_size]
  out = self.fc(out)
  return out
```

（4）TextRNN+Attertion

![img](README.assets/wps4.png)

```
def __init__(
  self,
  vocab_size,
  output_dim,
  n_layers=2,
  pad_idx=None,
  hidden_dim=128,
  hidden_size2=64,
  embed_dim=300,
  dropout=0.1,
  bidirectional=True,
):
  super().__init__()
  num_directions = 1 if not bidirectional else 2
  self.embedding = nn.Embedding(
    vocab_size,
    embed_dim,
    padding_idx=pad_idx,
  )
  self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
            bidirectional=True, batch_first=True, dropout=dropout)
  self.tanh1 = nn.Tanh()
  \# self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
  self.w = nn.Parameter(torch.zeros(hidden_dim * num_directions))
  self.tanh2 = nn.Tanh()
  self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
  self.fc = nn.Linear(hidden_size2, output_dim)
```

#### 3.超参数配置

（1）通用配置：

| train_batch_size | 64     | adam_epsilon     | 1    |
| ---------------- | ------ | ---------------- | ---- |
| eval_batch_size  | 64     | max_grad_norm    | 1.0  |
| num_labels       | 5      | num_train_epochs | 5    |
| vocab_size       | 400000 | warmup_steps     | 0    |
| dropout          | 0.1    | logging_steps    | 50   |
| learning_rate    | 1      | save_steps       | 100  |
| seed             | 66     |                  |      |

（2）TextCNN

```
def __init__(
  self,
  vocab_size, # 已知词的数量
  output_dim,  # 类别数
  pad_idx=None,
  embed_dim=300, # 每个词向量长度
  dropout=0.1,
  Ci=1,  ##输入的channel数
  kernel_num = 256, # 每种卷积核的数量
  kernel_sizes = [2,3,4], # 卷积核list，形如[2,3,4]
):
  super().__init__()

  Dim = embed_dim  ##每个词向量长度
  Cla = output_dim  ##类别数
  Ci = Ci  ##输入的channel数
  Knum = kernel_num  ## 每种卷积核的数量
  Ks = kernel_sizes  ## 卷积核list，形如[2,3,4]

  self.embedding = nn.Embedding(
    vocab_size,
    embed_dim,
    padding_idx=pad_idx,
  )

  self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
  self.dropout = nn.Dropout(dropout)
  self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层
```

（3）TextRNN

```
def __init__(
  self,
  vocab_size,
  output_dim,
  n_layers=2,
  pad_idx=None,
  hidden_dim=128,
  embed_dim=300,
  dropout=0.1,
  bidirectional=False,
):
  super().__init__()
  num_directions = 1 if not bidirectional else 2
  self.embedding = nn.Embedding(
    vocab_size,
    embed_dim,
    padding_idx=pad_idx,
  )
  self.lstm = nn.LSTM(
    embed_dim,
    hidden_dim,
    num_layers=n_layers,
    bidirectional=bidirectional,
    dropout=dropout,
  )
  self.dropout = nn.Dropout(p=dropout)
  self.linear = nn.Linear(hidden_dim * n_layers * num_directions,
              output_dim)
```

（4）TextRCNN

```
def __init__(
  self,
  vocab_size,
  output_dim,
  n_layers=2,
  pad_idx=None,
  hidden_dim=128,
  embed_dim=300,
  dropout=0.1,
):
  super().__init__()

  self.embedding = nn.Embedding(
    vocab_size,
    embed_dim,
    padding_idx=pad_idx,
  )

  self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
            bidirectional=True, batch_first=True, dropout=dropout)
  self.fc = nn.Linear(hidden_dim * 2 + embed_dim, output_dim)
```

（5）TextRNN+Attertion

```
def __init__(
  self,
  vocab_size,
  output_dim,
  n_layers=2,
  pad_idx=None,
  hidden_dim=128,
  hidden_size2=64,
  embed_dim=300,
  dropout=0.1,
  bidirectional=True,
):
  super().__init__()
  num_directions = 1 if not bidirectional else 2
  self.embedding = nn.Embedding(
    vocab_size,
    embed_dim,
    padding_idx=pad_idx,
  )
  self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
            bidirectional=True, batch_first=True, dropout=dropout)
  self.tanh1 = nn.Tanh()
  \# self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
  self.w = nn.Parameter(torch.zeros(hidden_dim * num_directions))
  self.tanh2 = nn.Tanh()
  self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
  self.fc = nn.Linear(hidden_size2, output_dim)
```

#### 4.评估方法

采用precesion、recall、f1_score作为评价指标，由于是多分类问题，采用微平均法，即precesion=recall=f1_score。

```
precision, recall, f1_score, _ = precision_recall_fscore_support(
  y_true, y_pred, average='micro')
```

#### 5. 测试集上的最终结果

| 模型                  | f1-score |
| --------------------- | -------- |
| TextCNN               | 0.6948   |
| TextRNN               | 0.7372   |
| TextRNN_bidirectional | 0.7692   |
| TextRNN+Attention     | 0.8112   |
| TextRCNN              | 0.8344   |

#### 6.tensorboard 可视化训练结果

（1）TextCNN

![img](README.assets/wps10.jpg)

（2）TextRNN

![img](README.assets/wps6.jpg) 

（3）TextRNN_bidirectional

![img](README.assets/wps7.jpg)

（4）TextRNN+Attention

![img](README.assets/wps8.jpg)

（5）TextRCNN

![img](README.assets/wps9.jpg)



## Features

- [ ] DPCNN 
- [ ] Transfromer