import torch
import torch.nn as nn
import torch.nn.functional as F


class TextSentimentCNN(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, 50))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        # 初始化权重
        for conv in self.convs:
            nn.init.normal_(conv.weight, mean=0, std=0.02)
            nn.init.constant_(conv.bias, 0)
        nn.init.normal_(self.fc.weight, mean=0, std=0.02)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, text: torch.Tensor):
        text = text.unsqueeze(1)  # [batch size, 1, sent len, emb dim]

        conveds = []
        for conv in self.convs:
            conved = F.relu(conv(text)).squeeze(3)
            pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            conveds.append(pooled)

        cat = self.dropout(torch.cat(conveds, dim=1))
        return self.fc(cat)


class TextSentimentRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(TextSentimentRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        nn.init.normal_(self.fc.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, text):
        # text shape: [batch size, seq len, emb dim] -> [64, 100, 50]
        output, (hidden, _) = self.lstm(text)
        hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
        hidden = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        hidden = self.fc(hidden)
        # hidden shape: [batch size, hidden_dim * num directions] -> [64, 256]
        return hidden  # [64, output_dim]


class TextSentimentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(TextSentimentMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, text):
        # text shape: [batch size, seq len, emb dim] -> [64, 100, 50]
        # 将输入的文本张量展平
        text = text.view(text.shape[0], -1)
        # text shape: [batch size, seq len * emb dim] -> [64, 5000]
        h1 = F.relu(self.fc1(text))
        h1 = self.dropout(h1)
        # h1 shape: [batch size, hidden dim] -> [64, 128]
        h2 = self.fc2(h1)
        return h2
