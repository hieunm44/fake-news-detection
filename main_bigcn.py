import argparse
import copy as cp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from utils.data_loader import DropEdge

torch.manual_seed(777)


"""
The Bi-GCN is adopted from the original implementation from the paper authors 

Paper: Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks
Link: https://arxiv.org/pdf/2001.06362.pdf
Source Code: https://github.com/TianBian95/BiGCN
"""


class TDrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(TDrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		# x: Size([7376, 300])
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)
		rootindex = data.root_index # torch.Size([128])
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device) # torch.Size([7376, 300])
		batch_size = max(data.batch) + 1 # 128

		for num_batch in range(batch_size): # num_batch=127
			index = (torch.eq(data.batch, num_batch)) # tensor([False, False, False,  ..., True, True, True])
			# root_index[num_batch]: 7275
			# x1[rootindex[num_batch]]: Size([300])
			root_extend[index] = x1[rootindex[num_batch]]

		x = torch.cat((x, root_extend), 1) # torch.Size([7376, 428])

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index) # torch.Size([7376, 128])
		x = F.relu(x)

		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device) # torch.Size([7376, 128])
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1) # torch.Size([7376, 256])

		x = scatter_mean(x, data.batch, dim=0) # torch.Size([128, 256]), lấy mean của x theo từng batch, có 128 batches

		return x


class BUrumorGCN(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(BUrumorGCN, self).__init__()
		self.conv1 = GCNConv(in_feats, hid_feats)
		self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

	def forward(self, data):
		x, edge_index = data.x, data.BU_edge_index
		x1 = cp.copy(x.float())
		x = self.conv1(x, edge_index)
		x2 = cp.copy(x)

		rootindex = data.root_index
		root_extend = torch.zeros(len(data.batch), x1.size(1)).to(rootindex.device)
		batch_size = max(data.batch) + 1
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x1[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1) # Size([7376, 428])

		x = F.relu(x)
		x = F.dropout(x, training=self.training)
		x = self.conv2(x, edge_index) # Size([7376, 128])
		x = F.relu(x)
		root_extend = torch.zeros(len(data.batch), x2.size(1)).to(rootindex.device) # Size([7376, 128])
		for num_batch in range(batch_size):
			index = (torch.eq(data.batch, num_batch))
			root_extend[index] = x2[rootindex[num_batch]]
		x = torch.cat((x, root_extend), 1) # Size([7376, 256])

		x = scatter_mean(x, data.batch, dim=0) # Size([128, 256])

		return x


class Net(torch.nn.Module):
	def __init__(self, in_feats, hid_feats, out_feats):
		super(Net, self).__init__()
		self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
		self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
		self.fc = torch.nn.Linear((out_feats+hid_feats) * 2, 2)

	def forward(self, data):
		TD_x = self.TDrumorGCN(data)
		BU_x = self.BUrumorGCN(data)
		x = torch.cat((TD_x, BU_x), 1)
		x = self.fc(x)
		x = F.log_softmax(x, dim=1)

		return x


def train(model, train_loader, optimizer):
	model.train()
	total_loss = 0

	for i, data in enumerate(train_loader): 
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
	parser.add_argument('--feature', type=str, default='profile', choices=['profile', 'spacy', 'bert', 'content'])
	parser.add_argument('--batch_size', type=int, default=128, help='batch size')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
	parser.add_argument('--d_hidden', type=int, default=128, help='hidden size')
	parser.add_argument('--n_epochs', type=int, default=100, help='maximum number of epochs')
	parser.add_argument('--n_early_stop', type=int, default=20, help='number of patience epochs in early stopping')
	parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
	parser.add_argument('--TDdroprate', type=float, default=0.2, help='dropout ratio')
	parser.add_argument('--BUdroprate', type=float, default=0.2, help='dropout ratio')
	args = parser.parse_args()

	train_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='train', transform=DropEdge(args.TDdroprate, args.BUdroprate))
	val_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='val', transform=DropEdge(args.TDdroprate, args.BUdroprate)) 
	test_dataset = UPFD(root=args.datapath, name=args.dataset, feature=args.feature, split='test', transform=DropEdge(args.TDdroprate, args.BUdroprate)) 

	train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) 
	val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net(in_feats=train_dataset.num_features, hid_feats=args.d_hidden, out_feats=args.d_hidden)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	run(args, model, train_loader, optimizer)