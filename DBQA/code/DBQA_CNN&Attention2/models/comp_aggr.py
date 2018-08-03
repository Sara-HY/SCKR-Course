import torch
from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()

    def forward(self, q, a):  # q: (Q, l), a: (N, A, l)
        s = q.matmul(a.transpose(1, 2))  # s: (N, Q, A)
        e = F.softmax(s, dim=1)  # e: (N, Q, A)
        h = e.transpose(1, 2).matmul(q)  # h: (N, A, l)
        return h


class CompareModule(nn.Module):
    def __init__(self):
        super(CompareModule, self).__init__()

    def forward(self, a, h):
        """
        :param a: (N, A, l)
        :param h: (N, A, l)
        :return:
        """
        out = a.mul(h)  # element-wise product. out: (N, A, l)
        return out


class ConvModule(nn.Module):
    def __init__(self, out_channel, kernel_sizes, emb_dim, class_num=1):
        super(ConvModule, self).__init__()
        self.out_channel = out_channel
        self.convs = nn.ModuleList([nn.Conv2d(1, out_channel, (ks, emb_dim)) for ks in kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_num)

    def forward(self, x):
        """
        :param x: (N, A, l)
        :return:
        """
        x = x.unsqueeze(1)  # x: (N, 1, A, l)

        # x: [(N, out_channel, X), ... ]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # x: [(N, out_channel), ...]
        x = torch.cat(x, 1)  # x: (N, len(kernel_size) * out_channel)

        x = F.dropout(x)
        x = self.fc(x)  # out: (N, class_num)
        return x


class CompAggrModel(BasicModule):
    def __init__(self, opt):
        super(CompAggrModel, self).__init__()
        self.module_name = 'comp_aggr_model'
        self.attention = AttentionModule()
        self.compare = CompareModule()
        self.conv = ConvModule(out_channel=opt.emb_dim, kernel_sizes=opt.kernel_sizes, emb_dim=opt.emb_dim)
        self.linear = nn.Linear(5, 1)

    def forward(self, q, a, s):
        h = self.attention(q, a)
        t = self.compare(a, h)
        r = self.conv(t)
        r = torch.cat([r, s], dim=1)
        score = self.linear(r)
        score = F.log_softmax(score.view(-1), dim=0)
        return score
