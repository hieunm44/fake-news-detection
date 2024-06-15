import argparse
from math import ceil

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import UPFD
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from sklearn.metrics import f1_score, accuracy_score

from utils.data_loader import ToUndirected


"""
The GNN-CL is implemented using DiffPool as the graph encoder and profile feature as the node feature 

Paper: Graph Neural Networks with Continual Learning for Fake News Detection from Social Media
Link: https://arxiv.org/pdf/2007.03316.pdf
"""


class GNN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels,
				 normalize=False, lin=True):
		super(GNN, self).__init__()
		self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
		self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
		self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
		self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
		self.bn3 = torch.nn.BatchNorm1d(out_channels)

		if lin is True:
			self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
									   out_channels)
		else:
			self.lin = None

	def bn(self, i, x):
		batch_size, num_nodes, num_channels = x.size()

		x = x.view(-1, num_channels)
		x = getattr(self, 'bn{}'.format(i))(x)
		x = x.view(batch_size, num_nodes, num_channels)
		return x

	def forward(self, x, adj, mask=None):
		batch_size, num_nodes, in_channels = x.size()

		x0 = x
		x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
		x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
		x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

		x = torch.cat([x1, x2, x3], dim=-1)

		if self.lin is not None:
			x = F.relu(self.lin(x))

		return x


class Net(torch.nn.Module):
	def __init__(self, in_channels=3, num_classes=6):
		super(Net, self).__init__()

		num_nodes = ceil(0.25 * max_nodes)
		self.gnn1_pool = GNN(in_channels, 64, num_nodes)
		self.gnn1_embed = GNN(in_channels, 64, 64, lin=False)

		num_nodes = ceil(0.25 * num_nodes)
		self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
		self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

		self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

		self.lin1 = torch.nn.Linear(3 * 64, 64)
		self.lin2 = torch.nn.Linear(64, num_classes)

	def forward(self, x, adj, mask=None):
		s = self.gnn1_pool(x, adj, mask)
		x = self.gnn1_embed(x, adj, mask)

		x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

		s = self.gnn2_pool(x, adj)
		x = self.gnn2_embed(x, adj)

		x, adj, l2, e2 = dense_diff_pool(x, adj, s)

		x = self.gnn3_embed(x, adj)

		x = x.mean(dim=1)
		x = F.relu(self.lin1(x))
		x = self.lin2(x)
		return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


def train(model, train_loader, optimizer):
	model.train()
	loss_all = 0
	for data in train_loader:
		data = data.to(device)
		optimizer.zero_grad()
		out, _, _ = model(data.x, data.adj, data.mask)
		loss = F.nll_loss(out, data.y.view(-1))
		loss.backward()
		loss_all += data.y.size(0) * loss.item()
		optimizer.step()
	return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    acc = 0
    f1 = 0

    total_correct = 0
    total_examples = 0
    for data in loader:
        data = data.to(device)
        pred, _, _ = model(data.x, data.adj, data.mask)
        pred = pred.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += len(data.y)

        acc += accuracy_score(data.y, pred) * len(data.y)
        f1 +=  f1_score(data.y, pred) * len(data.y)

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
	parser.add_argument('--feature', type=str, default='profile', choices=['profile', 'spacy', 'bert', 'content'])
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
	parser.add_argument('--n_epochs', type=int, default=1000, help='maximum number of epochs')
	parser.add_argument('--n_early_stop', type=int, default=20, help='number of patience epochs in early stopping')
	args = parser.parse_args()

	if args.dataset == 'politifact':
		max_nodes = 500
	else:
		max_nodes = 200 

	train_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='train', transform=T.ToDense(max_nodes), pre_transform=ToUndirected()) 
	val_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='val', transform=T.ToDense(max_nodes), pre_transform=ToUndirected()) 
	test_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='test', transform=T.ToDense(max_nodes), pre_transform=ToUndirected()) 

	train_loader = DenseDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DenseDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DenseDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net(in_channels=train_dataset.num_features, num_classes=train_dataset.num_classes).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	run(args, model, train_loader, optimizer)
