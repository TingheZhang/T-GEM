import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import csv
import scipy
import numpy as np

from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
import pickle as pl

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os, copy, sys
import random
from sklearn.model_selection import train_test_split

from torch.utils.checkpoint import checkpoint as cp
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, f1_score
import seaborn
import glob
from tqdm import tqdm

###########change the number to your GPU ID
########### if you are using CPU， then delete this number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
	"--head_num", default=None, type=int, required=True, help="The number of head for each layers"
)

# Other parameters
parser.add_argument(
	"--learning_rate",
	default=0.0001,
	type=float,
	help="learning rate used for training",
)
parser.add_argument(
	"--dropout_rate",
	default=0.3,
	type=float,
	help="dropout rate used for training",
)

parser.add_argument(
	"--act_fun",
	default='nan',
	type=str,
	help="The activation function at the model top layer, can be chosen from relu, leakyrelu, or gelu. Otherwise use nan for no activation function",
)
parser.add_argument(
	"--rand_seed",
	default=52,
	type=int,
	help="random seed used to split train test and val ",
)

parser.add_argument(
	"--batch_size",
	default=16,
	type=int,
	help="batch size  ",
)
parser.add_argument(
	"--epoch",
	default=50,
	type=int,
	help="how many epoch will be used for training ",
)

parser.add_argument(
	"--do_val", action="store_true", help="Whether compute attribution on validation set or test set"
)
parser.add_argument(
	"--abs", action="store_true", help="Whether to compute the absolute value of attribution score   "
)
parser.add_argument(
	"--result_dir",
	required=True,
	type=str,
	help="The dir used to  save result",
)
parser.add_argument(
	"--model_location",
	required=True,
	type=str,
	help="the location of model that is going to be intreperted",
)
parser.add_argument(
	"--attr_method",
	default='ig',
	type=str,
	help="choose which method to compute the attribution scores, choose from ig, sg_ig,vg_ig, gb, vg_gb,sg_gb ",
)

args = parser.parse_args()

##################
print(sys.argv)
d_ff = 1024
dropout_rate = args.dropout_rate
n_epochs = args.epoch
batch_size = args.batch_size

n_head = args.head_num
lr_rate = args.learning_rate
act_fun = args.act_fun
gain = 1

rand_state = args.rand_seed
n_gene = 1708
n_feature = 1708
# n_class=0
n_class = 34
query_gene = 64  # not using but cannot delete
val = args.do_val

attr_method = args.attr_method
save_memory = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class mulitiattention(torch.nn.Module):
	def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, mode):

		super(mulitiattention, self).__init__()
		self.n_head = n_head
		self.n_gene = n_gene
		self.batch_size = batch_size
		self.n_feature = n_feature
		self.mode = mode
		self.query_gene = query_gene

		self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
		self.WK = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
		self.WV = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)

		torch.nn.init.xavier_normal_(self.WQ, gain=gain)
		torch.nn.init.xavier_normal_(self.WK, gain=gain)
		torch.nn.init.xavier_normal_(self.WV)

		self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]), requires_grad=True)
		print('init')

	def QK_diff(self, Q_seq, K_seq):
		QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)

		return torch.nn.Softmax(dim=2)(QK_dif)

	def mask_softmax_self(self, x):

		d = x.shape[1]
		x = x * ((1 - torch.eye(d, d)).to(device))
		return x

	def attention(self, x, Q_seq, WK, WV):
		if self.mode == 0:
			K_seq = x * WK
			K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
			K_seq = K_seq.permute(0, 2, 1)
			V_seq = x * WV

			QK_product = Q_seq * K_seq
			z = torch.nn.Softmax(dim=2)(QK_product)

			z = self.mask_softmax_self(z)

			out_seq = torch.matmul(z, V_seq)
		########### not working right now
		if self.mode == 1:
			zz_list = []
			for q in range(self.n_gene // self.query_gene):
				K_seq = x * WK
				V_seq = x * WV
				Q_seq_x = x[:, (q * self.query_gene):((q + 1) * self.query_gene), :]
				Q_seq = Q_seq_x.expand(Q_seq_x.shape[0], Q_seq_x.shape[1], self.n_gene)
				K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.query_gene)
				K_seq = K_seq.permute(0, 2, 1)

				QK_diff = self.QK_diff(Q_seq, K_seq)

				z = torch.nn.Softmax(dim=2)(QK_diff)

				z = torch.matmul(z, V_seq)

				zz_list.append(z)
			out_seq = torch.cat(zz_list, dim=1)
		###########
		return out_seq

	def forward(self, x):

		x = torch.reshape(x, (x.shape[0], x.shape[1], 1))

		out_h = []
		for h in range(self.n_head):
			Q_seq = x * self.WQ[h, :, :]
			Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
			if save_memory:
				attention_out = cp(self.attention, x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
			else:
				attention_out = self.attention(x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])

			out_h.append(attention_out)
		out_seq = torch.cat(out_h, dim=2)
		out_seq = torch.matmul(out_seq, self.W_0)
		return out_seq


class layernorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(layernorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class res_connect(nn.Module):

	def __init__(self, size, dropout):
		super(res_connect, self).__init__()
		self.norm = layernorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, out):
		"Apply residual connection to any sublayer with the same size."
		return x + self.norm(self.dropout(out))


class MyNet(torch.nn.Module):
	def __init__(self, batch_size, n_head, n_gene, n_feature, n_class, query_gene, d_ff, dropout_rate, mode):
		super(MyNet, self).__init__()
		self.n_head = n_head
		self.n_gene = n_gene
		self.batch_size = batch_size
		self.n_feature = n_feature
		self.n_class = n_class
		self.d_ff = d_ff

		self.mulitiattention1 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
												mode)
		self.mulitiattention2 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
												mode)
		self.mulitiattention3 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
												mode)
		# self.fc=torch.nn.Linear(self.n_gene,1)
		self.fc = nn.Linear(self.n_gene, self.n_class)
		torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)
		self.ffn1 = nn.Linear(self.n_gene, self.d_ff)
		self.ffn2 = nn.Linear(self.d_ff, self.n_gene)
		self.dropout = nn.Dropout(dropout_rate)
		self.sublayer = res_connect(n_gene, dropout_rate)
		self.norm = layernorm(n_gene)

	def feedforward(self, x):
		out = F.relu(self.ffn1(x))
		out = self.ffn2(self.dropout(out))

		return out

	def forward(self, x):
		# gpu_tracker.track()
		out_attn = self.mulitiattention1(x)
		# out_attn = self.norm(out_attn)
		out_attn_1 = self.sublayer(x, out_attn)
		out_attn_2 = self.mulitiattention2(out_attn_1)
		out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
		out_attn_3 = self.mulitiattention3(out_attn_2)
		out_attn_3 = self.sublayer(out_attn_2, out_attn_3)

		if act_fun == 'relu':
			out_attn_3 = F.relu(out_attn_3)
		if act_fun == 'leakyrelu':
			m = torch.nn.LeakyReLU(0.1)
			out_attn_3 = m(out_attn_3)
		if act_fun == 'gelu':
			m = torch.nn.GELU()
			out_attn_3 = m(out_attn_3)
		y_pred = self.fc(out_attn_3)

		y_pred = F.log_softmax(y_pred, dim=1)

		return y_pred


def get_percent_acc(attr_sort_index, xx, yy, percent_to_be_0):
	acc = []

	X_test_input_percent = copy.deepcopy(xx)
	X_test_input_percent[:, attr_sort_index[0:int(len(attr_sort_index) * percent_to_be_0)]] = 0
	X_test_input_percent_tensor = torch.from_numpy(X_test_input_percent).to(device)
	model.eval()
	permutation_test = torch.randperm(X_test_input_percent_tensor.size()[0])
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, i in enumerate(range(0, X_test_input_percent_tensor.size()[0], batch_size)):
			indices = permutation_test[i:i + batch_size]

			batch_x_test, batch_y_test = X_test_input_percent_tensor[indices], torch.from_numpy(yy)[
				indices]
			batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
			output_test = model(batch_x_test.float())

			test_loss += F.nll_loss(output_test, batch_y_test, reduction='sum')

			pred = output_test.argmax(dim=1, keepdim=True)

			correct += pred.eq(batch_y_test.view_as(pred)).sum().item()

	test_loss /= len(X_test_input)
	acc.append(correct / len(X_test_input_percent_tensor))
	return acc


if __name__ == '__main__':

	y, data_df, pathway_gene, pathway, cancer_name = pl.load(open('pathway_data.pckl', 'rb'))
	data_ = np.array(data_df)
	x = np.float32(data_)
	gene_list = data_df.columns.tolist()

	x = np.float32(data_)

	encoder = LabelEncoder()
	y_label = encoder.fit_transform(y)
	class_label = np.unique(y)

	u, count = np.unique(y_label, return_counts=True)
	count_sort_ind = np.argsort(-count)
	y_label_unique_top34 = u[count_sort_ind[0:34]]

	x_top34 = []
	y_top34 = []
	sample_size = []

	for j, sample_label in enumerate(y_label_unique_top34):
		sample_index = np.argwhere(y_label == sample_label)[:, 0]
		sample_size.append(sample_index.shape)
		x_top34.append(x[sample_index])
		temp_y = y_label[sample_index]
		temp_y[temp_y == sample_label] = j
		y_top34.append(temp_y)

	X_train = []
	X_test = []
	y_train = []
	y_test = []

	for d in range(len(x_top34)):
		x_train, x_test, y_tr, y_te = train_test_split(x_top34[d], y_top34[d], test_size=0.2,
													   random_state=rand_state)
		X_train.append(x_train)
		X_test.append(x_test)
		y_train.append(y_tr)
		y_test.append(y_te)

	if val == True:

		X_train_val = []
		X_val = []
		y_train_val = []
		y_val = []

		for dd in range(len(x_top34)):
			x_train_val, x_val, y_tr_val, y_va = train_test_split(X_train[dd], y_train[dd], test_size=0.1,
																  random_state=rand_state)
			X_train_val.append(x_train_val)
			X_val.append(x_val)
			y_train_val.append(y_tr_val)
			y_val.append(y_va)

		X_train = X_train_val
		y_train = y_train_val
		X_val_input = torch.from_numpy(np.vstack(X_val))
		y_val_input = torch.from_numpy(np.hstack(y_val))

	X_train_input = torch.from_numpy(np.vstack(X_train))
	X_test_input = torch.from_numpy(np.vstack(X_test))
	y_train_input = torch.from_numpy(np.hstack(y_train))
	y_test_input = torch.from_numpy(np.hstack(y_test))

	file = args.model_location

	model = torch.load(file)
	model=model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08,
								 weight_decay=0, amsgrad=False)  ## val acc 98.75

	train_loss_list = []
	val_loss_list = []
	res = {}
	confusion_matrix_res = []
	mcc_res = []
	acc_res = []
	auc_res = []
	f1_res = []

	if val == True:
		permutation_val = torch.randperm(X_val_input.size()[0])

		correct_val = 0
		val_loss = 0

		model.eval()
		with torch.no_grad():
			batch_pred = []
			batch_y_val_list = []
			batch_pred_cate_val = []
			for batch_idx_val, i in enumerate(range(0, X_val_input.size()[0], batch_size)):
				indices_val = permutation_val[i:i + batch_size]
				batch_x_val, batch_y_val = X_val_input[indices_val], y_val_input[indices_val]

				batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)

				output_val = model(batch_x_val.float())
				val_loss += F.nll_loss(output_val, batch_y_val, reduction='sum')
				pred_val = output_val.argmax(dim=1, keepdim=True)

				correct_val += pred_val.eq(batch_y_val.view_as(pred_val)).sum().item()
				batch_pred.append(pred_val.cpu().data.numpy())
				batch_y_val_list.append(batch_y_val.cpu().data.numpy())
				batch_pred_cate_val.append(output_val.cpu().data.numpy())

			val_loss /= len(X_val_input)

			val_loss_list.append(val_loss)

			print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
				val_loss, correct_val, len(X_val_input),
				100. * correct_val / len(X_val_input)))

			yy_val = np.hstack(batch_y_val_list).reshape(-1, 1)
			ppred_classes = np.vstack(batch_pred)

			acc_val = accuracy_score(yy_val, ppred_classes)
			f1 = f1_score(yy_val, ppred_classes, average='micro')

			confusion_mat_val = metrics.confusion_matrix(yy_val, ppred_classes)
			mcc_val = metrics.matthews_corrcoef(yy_val, ppred_classes)

			encoder_ = LabelBinarizer()
			yy_val_ = encoder_.fit_transform(yy_val)
			roc_auc_val = metrics.roc_auc_score(yy_val_, np.exp(np.vstack(batch_pred_cate_val)), multi_class='ovr',
												average='micro')

	test_loss = 0
	correct = 0
	permutation_test = torch.randperm(X_test_input.size()[0])
	with torch.no_grad():
		batch_pred_test = []
		batch_y_test_list = []
		batch_pred_cate_test = []
		for batch_idx, i in enumerate(range(0, X_test_input.size()[0], batch_size)):
			indices = permutation_test[i:i + batch_size]
			batch_x_test, batch_y_test = X_test_input[indices], y_test_input[indices]
			batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
			output_test = model(batch_x_test.float())

			test_loss += F.nll_loss(output_test, batch_y_test, reduction='sum')

			pred = output_test.argmax(dim=1, keepdim=True)

			correct += pred.eq(batch_y_test.view_as(pred)).sum().item()
			batch_pred_test.append(pred.cpu().data.numpy())
			batch_y_test_list.append(batch_y_test.cpu().data.numpy())
			batch_pred_cate_test.append(output_test.cpu().data.numpy())

	test_loss /= len(X_test_input)
	acc_test = correct / len(X_test_input)

	yy_test = np.hstack(batch_y_test_list).reshape(-1, 1)
	pred_classes = np.vstack(batch_pred_test)

	acc_acc = accuracy_score(yy_test, pred_classes)
	f1 = f1_score(yy_test, pred_classes, average='micro')

	confusion_mat_test = metrics.confusion_matrix(yy_test, pred_classes)
	mcc_test = metrics.matthews_corrcoef(yy_test, pred_classes)

	encoder_ = LabelBinarizer()
	yy_test_ = encoder_.fit_transform(yy_test)
	roc_auc_test = metrics.roc_auc_score(yy_test_, np.exp(np.vstack(batch_pred_cate_test)), multi_class='ovr',
										average='micro')

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(X_test_input),
		100. * correct / len(X_test_input)))

	attr_ave_acc_mean_list = []
	attr_med_acc_mean_list = []
	attr_ave_acc_mean_list_10 = []
	attr_med_acc_mean_list_10 = []
	attr_ave_acc_mean_score_list = []
	attr_med_acc_mean_score_list = []
	attr_ave_acc_mean_score_list_10 = []
	attr_med_acc_mean_score_list_10 = []
	for cancer_type in tqdm(range(n_class)):
		if val == True:
			precision_val = confusion_mat_val[cancer_type, cancer_type] / sum(confusion_mat_val[cancer_type, :])
			precision_ = precision_val
			X_input = X_val  ## change this to X_val if you want to get the attribution for validation data
			y_input = y_val
		else:
			precision_test = confusion_mat_test[cancer_type, cancer_type] / sum(confusion_mat_test[cancer_type, :])
			precision_=precision_test
			X_input=X_test ## change this to X_val if you want to get the attribution for validation data
			y_input=y_test

		if attr_method == 'ig':
			from captum.attr import IntegratedGradients

			torch.manual_seed(123)
			np.random.seed(123)
			model.eval()
			ig = IntegratedGradients(model, multiply_by_inputs=True)

			permutation = torch.randperm(torch.from_numpy(X_input[cancer_type]).size()[0])

			n_correct, n_total = 0, 0
			attributions_list = []
			for batch_idx, i in enumerate(range(0, torch.from_numpy(X_input[cancer_type]).size()[0], 1)):
				indices = permutation[i:i + 1]
				batch_x, batch_y = torch.from_numpy(X_input[cancer_type])[indices], \
								   torch.from_numpy(X_input[cancer_type])[indices]
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				attributions, approximation_error = ig.attribute(batch_x, target=batch_y,
																 return_convergence_delta=True, baselines=0)
				attributions_list.append(attributions)

		if attr_method == 'sg_ig':
			from captum.attr import NoiseTunnel, IntegratedGradients, LayerConductance

			torch.manual_seed(123)
			np.random.seed(123)
			model.eval()
			ig = IntegratedGradients(model, multiply_by_inputs=True)
			nt = NoiseTunnel(ig)

			permutation = torch.randperm(torch.from_numpy(X_input[0]).size()[0])

			n_correct, n_total = 0, 0
			attributions_list = []
			for batch_idx, i in enumerate(range(0, torch.from_numpy(X_input[0]).size()[0], 1)):
				indices = permutation[i:i + 1]
				batch_x, batch_y = torch.from_numpy(X_input[0])[indices], \
								   torch.from_numpy(X_input[0])[indices]
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				attributions = nt.attribute(batch_x, target=batch_y, nt_type='smoothgrad_sq', baselines=0,
											stdevs=0.1)
				attributions_list.append(attributions)

		if attr_method == 'vg_ig':
			from captum.attr import NoiseTunnel, IntegratedGradients

			torch.manual_seed(123)
			np.random.seed(123)
			model.eval()
			ig = IntegratedGradients(model, multiply_by_inputs=True)
			nt = NoiseTunnel(ig)

			permutation = torch.randperm(torch.from_numpy(X_input[0]).size()[0])

			n_correct, n_total = 0, 0
			attributions_list = []
			for batch_idx, i in enumerate(range(0, torch.from_numpy(X_input[0]).size()[0], 1)):
				indices = permutation[i:i + 1]
				batch_x, batch_y = torch.from_numpy(X_input[0])[indices], torch.from_numpy(y_test[0])[indices]
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				attributions = nt.attribute(batch_x, target=batch_y, nt_type='vargrad', baselines=0, stdevs=0.1)
				attributions_list.append(attributions)

		if attr_method == 'gb':
			from captum.attr import GuidedBackprop

			torch.manual_seed(123)
			np.random.seed(123)
			model.eval()
			gb = GuidedBackprop(model)

			permutation = torch.randperm(torch.from_numpy(X_input[0]).size()[0])
			attributions_list = []
			for batch_idx, i in enumerate(range(0, torch.from_numpy(X_input[0]).size()[0], 1)):
				indices = permutation[i:i + 1]
				batch_x, batch_y = torch.from_numpy(X_input[0])[indices], torch.from_numpy(y_input[0])[indices]
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				attributions = gb.attribute(batch_x, target=batch_y)
				attributions_list.append(attributions)

		if attr_method == 'vg_gb':
			from captum.attr import GuidedBackprop, NoiseTunnel

			torch.manual_seed(123)
			np.random.seed(123)
			model.eval()
			gb = GuidedBackprop(model)
			nt = NoiseTunnel(gb)

			permutation = torch.randperm(torch.from_numpy(X_input[0]).size()[0])
			n_correct, n_total = 0, 0
			attributions_list = []
			for batch_idx, i in enumerate(range(0, torch.from_numpy(X_input[0]).size()[0], 1)):
				indices = permutation[i:i + 1]
				batch_x, batch_y = torch.from_numpy(X_input[0])[indices], torch.from_numpy(y_input[0])[indices]
				batch_x, batch_y = batch_x.to(device), batch_y.to(device)
				attributions = nt.attribute(batch_x, target=batch_y, nt_type='vargrad', stdevs=0.1)
				attributions_list.append(attributions)


		attributions_array = torch.cat(attributions_list).detach().cpu().numpy()
		attributions_array_ave = np.mean(attributions_array, axis=0)
		attributions_array_med = np.median(attributions_array, axis=0)

		attr_ave_sort_index = attributions_array_ave.argsort()[::-1]  # index from large to small
		attr_ave_acc_list = []

		for percent in np.linspace(0.005, 0.05, 10):
			attr_ave_acc_list.append(
				precision_ - np.array(
					get_percent_acc(attr_ave_sort_index, X_input[cancer_type], y_input[cancer_type], percent)))

		attr_ave_acc_mean = np.mean(attr_ave_acc_list)

		attr_med_sort_index = attributions_array_med.argsort()[::-1]  # index from large to small
		attr_med_acc_list = []

		# use top 5% to test the abliation result
		for percent in np.linspace(0.005, 0.05, 10):
			attr_med_acc_list.append(
				precision_ - np.array(
					get_percent_acc(attr_med_sort_index, X_input[cancer_type], y_input[cancer_type], percent)))
		attr_med_acc_mean = np.mean(attr_med_acc_list)

		if not os.path.exists(args.result_dir+ str(cancer_type)):
			os.makedirs(args.result_dir+ str(cancer_type), exist_ok=True)
		csvfile = open(args.result_dir + str(cancer_type) + '/result_l3_'+attr_method+'_5.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, attr_ave_acc_mean, attr_med_acc_mean,
							 attr_ave_acc_mean + precision_, attr_med_acc_mean + precision_,
							 np.array(attr_ave_acc_list), np.array(attr_med_acc_list)])
		else:
			writer.writerow([file, acc_test, attr_ave_acc_mean, attr_med_acc_mean,
							 attr_ave_acc_mean + precision_, attr_med_acc_mean + precision_,
							 np.array(attr_ave_acc_list), np.array(attr_med_acc_list)])
		csvfile.close()

		attr_ave_acc_mean_list.append(attr_ave_acc_mean)
		attr_ave_acc_mean_score_list.append(attr_ave_acc_mean + precision_)
		attr_med_acc_mean_list.append(attr_ave_acc_mean)
		attr_med_acc_mean_score_list.append(attr_med_acc_mean + precision_)

		# use top 10% to test the abliation result
		attr_ave_sort_index = attributions_array_ave.argsort()[::-1]  # index from large to small
		attr_ave_acc_list = []
		for percent in np.linspace(0.01, 0.1, 10):
			attr_ave_acc_list.append(
				precision_ - np.array(
					get_percent_acc(attr_ave_sort_index, X_input[cancer_type], y_input[cancer_type], percent)))

		attr_ave_acc_mean = np.mean(attr_ave_acc_list)

		attr_med_sort_index = attributions_array_med.argsort()[::-1]  # index from large to small
		attr_med_acc_list = []

		for percent in np.linspace(0.01, 0.1, 10):
			attr_med_acc_list.append(
				precision_ - np.array(
					get_percent_acc(attr_med_sort_index, X_input[cancer_type], y_input[cancer_type], percent)))
		attr_med_acc_mean = np.mean(attr_med_acc_list)

		csvfile = open(args.result_dir + str(cancer_type) + '/result_l3_'+attr_method+'_10.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, attr_ave_acc_mean, attr_med_acc_mean,
							 attr_ave_acc_mean + precision_, attr_med_acc_mean + precision_,
							 np.array(attr_ave_acc_list), np.array(attr_med_acc_list)])
		else:
			writer.writerow([file,  acc_test, attr_ave_acc_mean, attr_med_acc_mean,
							 attr_ave_acc_mean + precision_, attr_med_acc_mean + precision_,
							 np.array(attr_ave_acc_list), np.array(attr_med_acc_list)])
		csvfile.close()

		attr_ave_acc_mean_list_10.append(attr_ave_acc_mean)
		attr_ave_acc_mean_score_list_10.append(attr_ave_acc_mean + precision_)
		attr_med_acc_mean_list_10.append(attr_ave_acc_mean)
		attr_med_acc_mean_score_list_10.append(attr_med_acc_mean + precision_)

		# save the important input rank to GSEA fromat
		attr_ave_sort_index = attributions_array_ave.argsort()[::-1]  # index from large to small
		attr_ave_sort_gene = np.array(gene_list)[attr_ave_sort_index]
		attr_ave_sort_attr = attributions_array_ave[attr_ave_sort_index]

		if not os.path.exists(args.result_dir+'/GSEA/'+ str(cancer_type)):
			os.makedirs(args.result_dir+'/GSEA/'+ str(cancer_type), exist_ok=True)
		filename_base=args.result_dir+'/GSEA/'+ str(cancer_type)+'/head_' + str(n_head) + '_lr_' + str(lr_rate) + '_' + str(act_fun)
		np.savetxt(filename_base+'_ave_' + attr_method + '.txt',attr_ave_sort_gene, fmt='%s')
		np.savetxt(filename_base+ '_ave_' + attr_method + '.rnk',np.vstack((attr_ave_sort_gene, attr_ave_sort_attr)).T, fmt='%s', delimiter='\t')

		attr_med_sort_index = attributions_array_med.argsort()[::-1]  # index from large to small
		attr_med_sort_gene = np.array(gene_list)[attr_med_sort_index]
		np.savetxt(filename_base + '_med_' + attr_method + '.txt',
			attr_med_sort_gene, fmt='%s')
		attr_med_sort_attr = attributions_array_med[attr_med_sort_index]
		np.savetxt(filename_base + '_med_' + attr_method + '.rnk',
			np.vstack((attr_med_sort_gene, attr_med_sort_attr)).T, fmt='%s', delimiter='\t')
		if args.abs:
			# use the absolute value
			attr_ave_sort_index_abs = abs(attributions_array_ave).argsort()[::-1]  # index from large to small
			attr_ave_sort_gene_abs = np.array(gene_list)[attr_ave_sort_index_abs]
			attr_ave_sort_attr_abs = abs(attributions_array_ave)[attr_ave_sort_index_abs]
			if not os.path.exists(args.result_dir + '/GSEA/abs/' + str(cancer_type)):
				os.makedirs(args.result_dir + '/GSEA/abs/' + str(cancer_type), exist_ok=True)
			filename_base_abs = args.result_dir + '/GSEA/abs/' + str(cancer_type) + '/head_' + str(n_head) + '_lr_' + str(
				lr_rate) + '_' + str(act_fun)

			np.savetxt(filename_base_abs + '_ave_' + attr_method + '_abs.txt',
					   attr_ave_sort_gene_abs, fmt='%s')
			np.savetxt(filename_base_abs + '_ave_' + attr_method + '_abs.rnk',
					   np.vstack((attr_ave_sort_gene_abs, attr_ave_sort_attr_abs)).T, fmt='%s',
					   delimiter='\t')

			attr_med_sort_index_abs = abs(attributions_array_med).argsort()[::-1]  # index from large to small
			attr_med_sort_gene_abs = np.array(gene_list)[attr_med_sort_index_abs]
			np.savetxt(filename_base_abs + '_med_' + attr_method + '_abs.txt',
					   attr_med_sort_gene_abs, fmt='%s')
			attr_med_sort_attr_abs = abs(attributions_array_med)[attr_med_sort_index_abs]
			np.savetxt(filename_base_abs+ '_med_' + attr_method + '_abs.rnk',
					   np.vstack((attr_med_sort_gene_abs, attr_med_sort_attr_abs)).T, fmt='%s',
					   delimiter='\t')

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_ave.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_ave_acc_mean_list),
							 'detailed info for each cancer'] + attr_ave_acc_mean_list)
		else:
			writer.writerow([file, acc_test, np.sum(attr_ave_acc_mean_list),
							 'detailed info for each cancer'] + attr_ave_acc_mean_list)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_med.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_med_acc_mean_list),
							 'detailed info for each cancer'] + attr_med_acc_mean_list)
		else:
			writer.writerow([file,  acc_test, np.sum(attr_med_acc_mean_list),
							 'detailed info for each cancer'] + attr_med_acc_mean_list)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_ave_score.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_ave_acc_mean_score_list),
							 'detailed info for each cancer'] + attr_ave_acc_mean_score_list)
		else:
			writer.writerow([file, acc_test, np.sum(attr_ave_acc_mean_score_list),
							 'detailed info for each cancer'] + attr_ave_acc_mean_score_list)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_med_score.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_med_acc_mean_score_list),
							 'detailed info for each cancer'] + attr_med_acc_mean_score_list)
		else:
			writer.writerow([file, acc_test, np.sum(attr_med_acc_mean_score_list),
							 'detailed info for each cancer'] + attr_med_acc_mean_score_list)
		csvfile.close()
		# for 10%
		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_ave_10.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_ave_acc_mean_list_10),
							 'detailed info for each cancer'] + attr_ave_acc_mean_list_10)
		else:
			writer.writerow([file,  acc_test, np.sum(attr_ave_acc_mean_list_10),
							 'detailed info for each cancer'] + attr_ave_acc_mean_list_10)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_med_10.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_med_acc_mean_list_10),
							 'detailed info for each cancer'] + attr_med_acc_mean_list_10)
		else:
			writer.writerow([file, acc_test, np.sum(attr_med_acc_mean_list_10),
							 'detailed info for each cancer'] + attr_med_acc_mean_list_10)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_ave_score_10.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_ave_acc_mean_score_list_10),
							 'detailed info for each cancer'] + attr_ave_acc_mean_score_list_10)
		else:
			writer.writerow([file,  acc_test, np.sum(attr_ave_acc_mean_score_list_10),
							 'detailed info for each cancer'] + attr_ave_acc_mean_score_list_10)
		csvfile.close()

		csvfile = open(args.result_dir+'/result_l3_'+attr_method+'_allcancer_med_score_10.csv', 'at',
					   newline='')  # encoding='utf-8'
		writer = csv.writer(csvfile, delimiter=",")
		if val==True:
			writer.writerow([file, acc_val, acc_test, np.sum(attr_med_acc_mean_score_list_10),
							 'detailed info for each cancer'] + attr_med_acc_mean_score_list_10)
		else:
			writer.writerow([file,  acc_test, np.sum(attr_med_acc_mean_score_list_10),
							 'detailed info for each cancer'] + attr_med_acc_mean_score_list_10)
		csvfile.close()
