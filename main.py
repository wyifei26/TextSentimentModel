import json
import time

import torchtext
from torch import optim
from torch.utils.data import DataLoader

from loader import *
from models import *
from utils import *

torchtext.disable_torchtext_deprecation_warning()


@timeit
def train(model, iterator, optimizer, criterion):
    model.train()
    total_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    for texts, labels in iterator:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算混淆矩阵
        for i in range(len(outputs)):
            if outputs[i].argmax() == labels[i]:
                if labels[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if labels[i] == 1:
                    fn += 1
                else:
                    fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return total_loss / len(iterator), accuracy, f1


def evaluate(model, iterator, criterion):
    model.eval()
    total_loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with torch.no_grad():
        for texts, labels in iterator:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 计算混淆矩阵
            for i in range(len(outputs)):
                if outputs[i].argmax() == labels[i]:
                    if labels[i] == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if labels[i] == 1:
                        fn += 1
                    else:
                        fp += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return total_loss / len(iterator), accuracy, f1


@timeit
def main(model_name='CNN', n_filters=32, filter_sizes=[3, 4, 5], lr=0.001, seq_len=100,
         output_dim=2, dropout=0.5, batch_size=32, epochs=10, n_layers=2, hidden_dim=128):
    start = time.time()

    # 预处理数据
    word_dict, word_vectors, test_dataset, train_dataset, valid_dataset = preprocess(seq_len)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, collate_fn=collate_batch, shuffle=True)

    # 创建模型
    if model_name == 'RNN':
        model = TextSentimentRNN(word_vectors.shape[1], hidden_dim, output_dim, n_layers, dropout)
    elif model_name == 'CNN':
        model = TextSentimentCNN(n_filters, filter_sizes, output_dim, dropout)
    elif model_name == 'MLP':
        model = TextSentimentMLP(word_vectors.shape[1] * seq_len, hidden_dim, output_dim, dropout)
    else:
        raise ValueError('Invalid model name')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练和评估
    training_data = {'train_loss': [], 'train_acc': [], 'train_f1': [], 'valid_loss': [], 'valid_acc': [],
                     'valid_f1': []}
    for epoch in range(epochs):
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion)
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}, Train F1: {train_f1:.2f},'
              f' Valid Loss: {valid_loss:.3f}, Valid Acc: {valid_acc:.2f}, Valid F1: {valid_f1:.2f}')
        training_data['train_loss'].append(train_loss)
        training_data['train_acc'].append(train_acc)
        training_data['valid_loss'].append(valid_loss)
        training_data['valid_acc'].append(valid_acc)
        training_data['train_f1'].append(train_f1)
        training_data['valid_f1'].append(valid_f1)

    # 测试
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    training_data['test_acc'] = test_acc
    training_data['test_f1'] = test_f1
    training_data['test_loss'] = test_loss
    print(f'Test Acc: {test_acc:.2f}')

    # 保存训练数据
    training_data['time'] = time.time() - start
    with open(f'./logs/{model_name}-lr{lr}.json', 'w') as f:
        json.dump(training_data, f, indent=4)


if __name__ == "__main__":
    main()
