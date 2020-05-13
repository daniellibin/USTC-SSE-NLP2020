import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import copy


class DPCNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        output_dim,
        pad_idx=None,
        dropout=0.1,
        embed_dim=300,
        kernel_num=250,
    ):
        super().__init__()


        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )

        self.conv_region = nn.Conv2d(1, kernel_num, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(kernel_num, kernel_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(kernel_num, output_dim)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 0)  # 将 0轴 和 1轴 ， # [batch_size, seq_len, embed_size]

        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_size]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] >= 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x) # [batch_size, 250, seq_len-3+1 +1, 1] = [batch_size, 250, seq_len-1, 1]
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

class TextRCNN(nn.Module):

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

class TextCNN(nn.Module):

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

class TextRNN(nn.Module):

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

    def forward(self, x, x_len):
        x = self.embedding(x)
        # Pad each sentences for a batch,
        # the final x with shape (seq_len, batch_size, embed_size)
        x = pack_padded_sequence(x, x_len)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        # NOTE: take the last hidden state of encoder as in seq2seq architecture.
        '''
        output保存了最后一层，每个time step的输出h，如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)。
        h_n保存了每一层，最后一个time step的输出h，如果是双向LSTM，单独保存前向和后向的最后一个time step的输出h。
        c_n与h_n一致，只是它保存的是cell的值。
        '''
        hidden_states, (h_n, c_c) = self.lstm(x)
        h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous() # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        # h_n:(batch_size, hidden_size * num_layers * num_directions)
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.linear(h_n)
        return loggits

class TextRNN_Att(nn.Module):

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
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(hidden_dim * num_directions))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_dim)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 0)  # 将 0轴 和 1轴 交换

        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # self.w 维度为[256,1]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1],dim=1是对seq_len即每个时刻归一化
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # 对seq_len进行求和，变为[128, 256]
        out = F.relu(out)
        out = self.fc1(out)  # [128, 256] * [286,64] -->[128,64]
        out = self.fc(out)  # [128, 64] * [64,5] --> [128, 5]
        return out

class Transfromer(nn.Module):

    def __init__(
        self,
        vocab_size,
        output_dim,
        pad_idx=None,
        hidden=1024,
        embed_dim=300,
        dim_model = 300,
        dropout=0.5,
        bidirectional=True,
        device = 'cpu',
        num_head = 5,
        num_encoder = 2
    ):
        super().__init__()
        num_directions = 1 if not bidirectional else 2
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.device = device
        self.dim_model = dim_model
        self.output_dim = output_dim

        self.encoder = Encoder(dim_model, num_head, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(num_encoder)])

        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x, x_len):
        out = self.embedding(x)
        pad_size = x_len[0].item()
        out = torch.transpose(out, 1, 0)  # 将 0轴 和 1轴 交换

        postion_embedding = Positional_Encoding(self.embed_dim, pad_size, self.dropout, self.device)
        out = postion_embedding(out)

        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)

        fc1 = nn.Linear(pad_size * self.dim_model, self.output_dim)
        out = fc1(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
