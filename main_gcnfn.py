import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import global_mean_pool, GATConv
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

torch.manual_seed(777)


"""
GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder; 
the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional 
comment word2vec (spaCy) embeddings plus 10-dimensional profile features 

Paper: Fake News Detection on Social Media using Geometric Deep Learning
Link: https://arxiv.org/pdf/1902.06673.pdf


Model Configurations:

Vanilla GCNFN: args.concat = False, args.feature = content
UPFD-GCNFN: args.concat = True, args.feature = spacy
"""


class Net(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, concat=False):
		super(Net, self).__init__()
            
		self.concat = concat

		self.conv1 = GATConv(in_channels, hidden_channels * 2)
		self.conv2 = GATConv(hidden_channels * 2, hidden_channels * 2)

		self.fc1 = Linear(hidden_channels * 2, hidden_channels)

		if self.concat:
			self.fc0 = Linear(in_channels, hidden_channels)
			self.fc1 = Linear(hidden_channels * 2, hidden_channels)

		self.fc2 = Linear(hidden_channels, out_channels)


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = F.selu(self.conv1(x, edge_index))
		x = F.selu(self.conv2(x, edge_index))
		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.fc0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.fc1(x))

		x = F.log_softmax(self.fc2(x), dim=-1)

		return x


def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader: 
        data = data.to(device)
        optimizer.zero_grad()                    
        out = model(data)

        loss = F.nll_loss(out, data.y) 
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    acc = 0
    f1 = 0

    total_correct = 0
    total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

        acc += accuracy_score(data.y, pred) * data.num_graphs
        f1 +=  f1_score(data.y, pred) * data.num_graphs

    return acc / total_examples, f1 / total_examples


def run(args, model, train_loader, optimizer):
    best_acc = -1
    count = 0
    for epoch in range(args.n_epochs):
        loss = train(model, train_loader, optimizer)
        train_acc, train_f1 = test(train_loader)
        val_acc, val_f1 = test(val_loader)
        test_acc, test_f1 = test(test_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch
            count = 0
        else:
            count +=1
        
        if count > args.n_early_stop:
            break
        
        print(f'epoch: {epoch:02d}, loss: {loss:.4f} ',
              f'train_acc: {train_acc:.4f}, train_f1: {train_f1:.4f} '
              f'val_acc: {val_acc:.4f}, val_f1: {val_f1:.4f} ',
              f'test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}')
    
    print(f'best epoch: {best_epoch}, best acc: {best_acc}, best_f1: {best_f1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'])
    parser.add_argument('--datapath', type=str, default='data', help='data folder')
    parser.add_argument('--feature', type=str, default='spacy', choices=['profile', 'spacy', 'bert', 'content'])
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--d_hidden', type=int, default=128, help='hidden size')
    parser.add_argument('--n_epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--n_early_stop', type=int, default=20, help='number of patience epochs in early stopping')
    parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
    args = parser.parse_args()

    train_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='train', transform=ToUndirected()) 
    val_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='val', transform=ToUndirected()) 
    test_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='test', transform=ToUndirected()) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(train_dataset.num_features, args.d_hidden, train_dataset.num_classes, concat=args.concat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run(args, model, train_loader, optimizer)