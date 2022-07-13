import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import glob
import csv
import scipy
import numpy as np

from sklearn.preprocessing import LabelBinarizer, LabelEncoder,OneHotEncoder
import pickle as pl

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os,copy,sys
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools

import inspect
# import torch.utils.checkpoint as cp
from torch.utils.checkpoint import checkpoint as cp
from sklearn import svm,metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
import seaborn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.set_num_threads(10)
print(sys.argv)
d_ff = 1024
dropout_rate = 0.3
n_epochs = 50
batch_size = 64
n_head = 5
gain=5
lr_rate=0.0005
# rand_state=np.int(sys.argv[4])
act_fun='nan'
rand_state=52

n_gene = 1708
n_feature = 1708
# n_class=34
n_class = 34
query_gene = 64
val = True

#######choose the function at the begining,
#######'attn' for get the attn value for each layer, 'vis' for viulize the attn
fun=sys.argv[1]

save_memory = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class mulitiattention(torch.nn.Module):
	def __init__(self, batch_size,n_head,n_gene,n_feature,query_gene,mode):


		super(mulitiattention, self).__init__()
		self.n_head=n_head
		self.n_gene = n_gene
		self.batch_size=batch_size
		self.n_feature=n_feature
		self.mode=mode
		self.query_gene=query_gene

		self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
		self.WK = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)
		self.WV = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)

		torch.nn.init.xavier_normal_(self.WQ,gain=gain)
		torch.nn.init.xavier_normal_(self.WK,gain=gain)

		torch.nn.init.xavier_normal_(self.WV)

		self.W_0=nn.Parameter(torch.Tensor(self.n_head*[1]),requires_grad=True)


		print('init')

	def QK_diff(self,Q_seq, K_seq):
		QK_dif = -1 * torch.pow((Q_seq - K_seq),2)

		return torch.nn.Softmax(dim=2)(QK_dif)

	def mask_softmax_self(self,x):

		d=x.shape[1]
		x = x *((1 - torch.eye(d, d)).to(device))
		return x

	def attention(self,x,Q_seq,WK,WV):
		if self.mode == 0:
			K_seq = x * WK
			K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
			K_seq = K_seq.permute(0, 2, 1)
			V_seq = x * WV

			QK_product = Q_seq * K_seq
			z=torch.nn.Softmax(dim=2)(QK_product)

			z=self.mask_softmax_self(z)

			out_seq=torch.matmul(z, V_seq)

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
		return out_seq

	def forward(self, x):

		x = torch.reshape(x, (x.shape[0], x.shape[1], 1))

		out_h = []
		for h in range(self.n_head):
			Q_seq = x * self.WQ[h,:,:]
			Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
			if save_memory:
				attention_out=cp(self.attention,x, Q_seq, self.WK[h,:,:], self.WV[h,:,:])
			else:
				attention_out=self.attention(x, Q_seq, self.WK[h,:,:], self.WV[h,:,:])

			out_h.append(attention_out)
		out_seq=torch.cat(out_h,dim=2)
		out_seq=torch.matmul(out_seq,self.W_0)
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
	def __init__(self, batch_size,n_head,n_gene,n_feature,n_class,query_gene,d_ff,dropout_rate,mode):
		super(MyNet, self).__init__()
		self.n_head=n_head
		self.n_gene = n_gene
		self.batch_size=batch_size
		self.n_feature=n_feature
		self.n_class=n_class
		self.d_ff=d_ff

		self.mulitiattention1=mulitiattention(self.batch_size,self.n_head,self.n_gene,self.n_feature,query_gene,mode)
		self.mulitiattention2 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
											   mode)
		self.mulitiattention3 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
											   mode)

		self.fc = nn.Linear(self.n_gene, self.n_class)
		torch.nn.init.xavier_uniform_(self.fc.weight,gain=1)
		self.ffn1=nn.Linear(self.n_gene, self.d_ff)
		self.ffn2 = nn.Linear(self.d_ff,self.n_gene)
		self.dropout=nn.Dropout(dropout_rate)
		self.sublayer=res_connect(n_gene,dropout_rate)
		self.norm = layernorm(n_gene)

	def feedforward(self,x):
		out=F.relu(self.ffn1(x))
		out=self.ffn2(self.dropout(out))

		return out

	def forward(self, x):

		out_attn= self.mulitiattention1(x)

		out_attn_1=self.sublayer(x,out_attn)
		out_attn_2 = self.mulitiattention2(out_attn_1)
		out_attn_2=self.sublayer(out_attn_1,out_attn_2)
		out_attn_3 = self.mulitiattention3(out_attn_2)
		out_attn_3=self.sublayer(out_attn_2,out_attn_3)

		if act_fun=='relu':
			out_attn_3=F.relu(out_attn_3)
		if act_fun=='leakyrelu':
			m=torch.nn.LeakyReLU(0.1)
			out_attn_3=m(out_attn_3)
		if act_fun == 'gelu':
			m = torch.nn.GELU()
			out_attn_3 = m(out_attn_3)
		y_pred = self.fc(out_attn_3)

		y_pred=F.log_softmax(y_pred, dim=1)

		return y_pred

if __name__ == '__main__':

	y, data_df, pathway_gene, pathway, cancer_name = pl.load(open('../pathway_data.pckl', 'rb'))
	data_ = np.array(data_df)
	x = np.float32(data_)
	gene_list = data_df.columns.tolist()

	encoder=LabelEncoder()
	y_label= encoder.fit_transform(y)
	class_label=np.unique(y)

	u, count = np.unique(y_label,return_counts=True)
	count_sort_ind = np.argsort(-count)
	y_label_unique_top34=u[count_sort_ind[0:34]]

	x_top34=[]
	y_top34=[]
	sample_size=[]

	for j, sample_label in enumerate(y_label_unique_top34):
		sample_index=np.argwhere(y_label==sample_label)[:,0]
		sample_size.append(sample_index.shape)
		x_top34.append(x[sample_index])
		temp_y=y_label[sample_index]
		temp_y[temp_y == sample_label] = j
		y_top34.append(temp_y)

	X_train=[]
	X_test=[]
	y_train=[]
	y_test=[]


	for d in range(len(x_top34)):
		x_train, x_test, y_tr, y_te = train_test_split(x_top34[d], y_top34[d], test_size=0.2,
															random_state=rand_state)
		X_train.append(x_train)
		X_test.append(x_test)
		y_train.append(y_tr)
		y_test.append(y_te)


	if val==True:

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

		X_train=X_train_val
		y_train=y_train_val
		X_val_input=torch.from_numpy(np.vstack(X_val))
		y_val_input=torch.from_numpy(np.hstack(y_val))


	X_train_input=torch.from_numpy(np.vstack(X_train))
	X_test_input=torch.from_numpy(np.vstack(X_test))
	y_train_input=torch.from_numpy(np.hstack(y_train))
	y_test_input=torch.from_numpy(np.hstack(y_test))

	model=MyNet(batch_size,n_head,n_gene,n_feature,n_class,query_gene,d_ff,dropout_rate,mode=0).to(device)

	file = glob.glob(
		'../model_test/pytorch_transformer_head*_epoch*')
	model = torch.load(file[0])

	def get_softmax_map(model_WQ,model_WK, model_WV,model_W0, input_gene):
		def mask_softmax_self(x):

			d = x.shape[1]
			x = x * ((1 - torch.eye(d, d)).to(device))
			return x
		input_gene = input_gene.view(-1, n_gene, 1)
		out_seq_z=[]
		softmax_p_list=[]
		for ind_ in range(n_head):
			input_K = input_gene * model_WK[ind_]
			input_Q = input_gene * model_WQ[ind_]
			K_seq = input_K.expand(input_K.shape[0], input_K.shape[1], n_gene)
			Q_seq = input_Q.expand(input_Q.shape[0], input_Q.shape[1], n_gene)
			K_seq = K_seq.permute(0, 2, 1)
			V_seq = input_gene * model_WV[ind_]
			QK_product = Q_seq * K_seq
			softmax_p = torch.nn.Softmax(dim=2)(QK_product)
			softmax_p_list.append(softmax_p)

			z = mask_softmax_self(softmax_p)

			out_seq_z.append(torch.matmul(z, V_seq))

		out_seq_z_cat=torch.cat(out_seq_z,dim=2)
		out_seq=torch.matmul(out_seq_z_cat, model_W0)

		return softmax_p_list, out_seq,out_seq_z_cat

	def get_softmax_map_single(model_WQ,model_WK, model_WV, input_gene,query_gene):
		input_gene=torch.from_numpy(input_gene).to(device)
		query_gene=torch.tensor(query_gene).to(device)
		input_gene = input_gene.view(-1, 4096, 1)

		input_K = input_gene * model_WK
		input_Q = input_gene * model_WQ
		K_seq = input_K
		Q_seq = input_Q.expand(input_Q.shape[0], input_Q.shape[1], n_gene)

		K_seq = K_seq.permute(0, 2, 1)
		V_seq = input_gene * model_WV

		QK_product = Q_seq * K_seq
		softmax_p = torch.nn.Softmax(dim=2)(QK_product)
		QK_dif_K = -1 * torch.pow((query_gene - K_seq), 2)
		QK_dif = -1 * torch.pow((query_gene - input_gene.permute(0, 2, 1)), 2)
		softmax_p_K = torch.nn.Softmax(dim=2)(QK_dif_K)
		softmax_p = torch.nn.Softmax(dim=2)(QK_dif)

		out_seq = torch.matmul(softmax_p_K, V_seq)
		return softmax_p_K, out_seq,softmax_p

	def compute_saliency_maps(X, y, model):

		X_var = Variable(X, requires_grad=True)

		torch.set_grad_enabled(True)

		logits = model(X_var)
		logits = logits.gather(1, y.view(-1, 1)).squeeze()  
		logits.backward(torch.FloatTensor([1.0]*logits.shape[0]).to(device)) 

		saliency = abs(X_var.grad.data) 
		return saliency.squeeze()


	model = model.to(device)
	model.eval()

	model_WQ1 = model.mulitiattention1.WQ
	model_WK1 = model.mulitiattention1.WK
	model_WV1 = model.mulitiattention1.WV
	model_W01 = model.mulitiattention1.W_0
	model_WK1.requires_grad = False
	model_WV1.requires_grad = False
	model_W01.requires_grad = False

	model_WQ2 = model.mulitiattention2.WQ
	model_WK2 = model.mulitiattention2.WK
	model_WV2 = model.mulitiattention2.WV
	model_W02 = model.mulitiattention2.W_0
	model_WK2.requires_grad = False
	model_WV2.requires_grad = False
	model_W02.requires_grad = False

	model_WQ3 = model.mulitiattention3.WQ
	model_WK3 = model.mulitiattention3.WK
	model_WV3 = model.mulitiattention3.WV
	model_W03 = model.mulitiattention3.W_0
	model_WK3.requires_grad = False
	model_WV3.requires_grad = False
	model_W03.requires_grad = False


	# ###################get the performance from the each attention layer or head
	####train
	if fun=='attn':
		permutation_test = torch.randperm(X_train_input.size()[0])
		attn_1=[]
		attn_2=[]
		attn_3=[]
		attn_1_sub=[]
		attn_2_sub=[]
		attn_3_sub=[]

		head_output_1=[]
		head_output_2=[]
		head_output_3=[]

		layer_entropy1 = []
		layer_entropy2 = []
		layer_entropy3 = []

		train_softmax_p_1 = []
		train_softmax_p_1_entropy = []
		# train_softmax_p_1_avg = 0
		train_softmax_p_1_avg = []

		train_softmax_p_2 = []
		# train_softmax_p_2_avg = 0
		train_softmax_p_2_avg=[]
		train_softmax_p_2_entropy = []

		train_softmax_p_3 = []
		train_softmax_p_3_entropy = []
		# train_softmax_p_3_avg = 0
		train_softmax_p_3_avg=[]

		y_train_attn=[]
		correct_train=0
		n_samples = len(X_train_input)
		with torch.no_grad():
			model.eval()

			for batch_idx, i in enumerate(range(0, X_train_input.size()[0], batch_size)):

				indices = permutation_test[i:i + batch_size]
				batch_x_train, batch_y_train = X_train_input[indices], y_train_input[indices]
				batch_x_train, batch_y_train = batch_x_train.to(device), batch_y_train.to(device)
				y_train_attn.append(batch_y_train.cpu().data.numpy())
				print('train ' + str(len(indices)/X_train_input.size()[0]*batch_idx))

				train_softmax_p1, attn1,head_output1 = get_softmax_map(model_WQ1,model_WK1, model_WV1,model_W01, batch_x_train)
				attn_1.append(attn1.cpu().data.numpy())
				head_output_1.append(head_output1.cpu().data.numpy())
				attn1 = attn1.view(-1, attn1.shape[1])

				attn1_ = model.sublayer(batch_x_train, attn1)
				attn_1_sub.append(attn1_.cpu().data.numpy())
				train_softmax_p2, attn2,head_output2 = get_softmax_map(model_WQ2,model_WK2, model_WV2,model_W02, attn1_)
				attn_2.append(attn2.cpu().data.numpy())
				head_output_2.append(head_output2.cpu().data.numpy())
				attn2 = attn2.view(-1, attn2.shape[1])

				attn2_ = model.sublayer(attn1_, attn2)
				attn_2_sub.append(attn2_.cpu().data.numpy())
				train_softmax_p3, attn3,head_output3 = get_softmax_map(model_WQ3,model_WK3, model_WV3,model_W03, attn2_)
				attn_3.append(attn3.cpu().data.numpy())
				head_output_3.append(head_output3.cpu().data.numpy())
				attn3 = attn3.view(-1, attn3.shape[1])

				attn3_ = model.sublayer(attn2_, attn3)
				attn_3_sub.append(attn3_.cpu().data.numpy())

				y_pred = model.fc(attn3_)
				pred_train = F.log_softmax(y_pred, dim=1).argmax(dim=1, keepdim=True)

				correct_train += pred_train.eq(batch_y_train.view_as(pred_train)).sum().item()

				layer_entropy1_h = []
				train_softmax_p_1_h_avg = []
				train_softmax_p_1_h_entropy = []
				layer_entropy2_h = []
				train_softmax_p_2_h_avg = []
				train_softmax_p_2_h_entropy = []
				train_softmax_p_3_h_entropy = []
				layer_entropy3_h = []
				train_softmax_p_3_h_avg = []
				for ind_h in range(n_head):
					layer_entropy1_h.append(scipy.stats.entropy(train_softmax_p1[ind_h].cpu().data.numpy(), axis=2))

					train_softmax_p_1_h_avg.append(np.sum(train_softmax_p1[ind_h].cpu().data.numpy(), axis=0))
					train_softmax_p_1_h_entropy.append(scipy.stats.entropy(train_softmax_p1[ind_h].cpu().data.numpy(), axis=2))

					train_softmax_p_2_h_entropy.append(scipy.stats.entropy(train_softmax_p2[ind_h].cpu().data.numpy(), axis=2))
					layer_entropy2_h.append(scipy.stats.entropy(train_softmax_p2[ind_h].cpu().data.numpy(), axis=2))

					train_softmax_p_2_h_avg.append(np.sum(train_softmax_p2[ind_h].cpu().data.numpy(), axis=0))
					train_softmax_p_3_h_entropy.append(scipy.stats.entropy(train_softmax_p3[ind_h].cpu().data.numpy(), axis=2))
					layer_entropy3_h.append(scipy.stats.entropy(train_softmax_p3[ind_h].cpu().data.numpy(), axis=2))

					train_softmax_p_3_h_avg.append(np.sum(train_softmax_p3[ind_h].cpu().data.numpy(), axis=0))

				layer_entropy1.append(layer_entropy1_h)
				layer_entropy2.append(layer_entropy2_h)
				layer_entropy3.append(layer_entropy3_h)
				train_softmax_p_1_avg.append(train_softmax_p_1_h_avg)
				train_softmax_p_2_avg.append(train_softmax_p_2_h_avg)
				train_softmax_p_3_avg.append(train_softmax_p_3_h_avg)
				train_softmax_p_1_entropy.append(train_softmax_p_1_h_entropy)
				train_softmax_p_2_entropy.append(train_softmax_p_2_h_entropy)
				train_softmax_p_3_entropy.append(train_softmax_p_3_h_entropy)

		print(correct_train / len(X_train_input))
		attn_1_train=np.vstack(attn_1)
		attn_2_train=np.vstack(attn_2)
		attn_3_train=np.vstack(attn_3)
		attn_1_sub_train=np.vstack(attn_1_sub)
		attn_2_sub_train=np.vstack(attn_2_sub)
		attn_3_sub_train=np.vstack(attn_3_sub)
		head_output_1_train = np.vstack(head_output_1)
		head_output_2_train = np.vstack(head_output_2)
		head_output_3_train=np.vstack(head_output_3)

		train_layer_entropy1_all=np.concatenate(layer_entropy1, axis=1)
		train_layer_entropy2_all = np.concatenate(layer_entropy2, axis=1)
		train_layer_entropy3_all = np.concatenate(layer_entropy3, axis=1)
		train_softmax_p_1_avg_all=np.sum(train_softmax_p_1_avg, axis=0)
		train_softmax_p_2_avg_all=np.sum(train_softmax_p_2_avg, axis=0)
		train_softmax_p_3_avg_all=np.sum(train_softmax_p_3_avg, axis=0)
		train_softmax_p_1_entropy_all=np.concatenate(train_softmax_p_1_entropy, axis=1)
		train_softmax_p_2_entropy_all = np.concatenate(train_softmax_p_2_entropy, axis=1)
		train_softmax_p_3_entropy_all = np.concatenate(train_softmax_p_3_entropy, axis=1)
		train_softmax_p_ave = np.stack((train_softmax_p_1_avg_all / n_samples, train_softmax_p_2_avg_all / n_samples,
									   train_softmax_p_3_avg_all / n_samples))

		y_train_attn=np.hstack(y_train_attn)
		###test
		permutation_test = torch.randperm(X_test_input.size()[0])
		output_z = []
		test_y = []
		output_softmax_p = []
		output_attn = []
		output_sal = []
		attn_1=[]
		attn_2=[]
		attn_3=[]
		attn_1_sub=[]
		attn_2_sub=[]
		attn_3_sub=[]

		head_output_1=[]
		head_output_2=[]
		head_output_3=[]

		layer_entropy1 = []
		layer_entropy2 = []
		layer_entropy3 = []

		test_softmax_p_1 = []
		test_softmax_p_1_entropy = []
		# test__softmax_p_1_avg = 0
		test_softmax_p_1_avg = []

		test_softmax_p_2 = []
		# test__softmax_p_2_avg = 0
		test_softmax_p_2_avg=[]
		test_softmax_p_2_entropy = []

		test_softmax_p_3 = []
		test_softmax_p_3_entropy = []
		# test__softmax_p_3_avg = 0
		test_softmax_p_3_avg=[]
		n_samples = len(X_test_input)
		y_test_attn=[]
		correct_test=0
		with torch.no_grad():
			model.eval()

			for batch_idx, i in enumerate(range(0, X_test_input.size()[0], batch_size)):
				print('test '+str(batch_idx))
				indices = permutation_test[i:i + batch_size]
				batch_x_test, batch_y_test = X_test_input[indices], y_test_input[indices]
				batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
				y_test_attn.append(batch_y_test.cpu().data.numpy())

				test_softmax_p1, attn1,head_output1 = get_softmax_map(model_WQ1,model_WK1, model_WV1,model_W01, batch_x_test)
				attn_1.append(attn1.cpu().data.numpy())
				head_output_1.append(head_output1.cpu().data.numpy())
				attn1 = attn1.view(-1, attn1.shape[1])

				attn1_ = model.sublayer(batch_x_test, attn1)
				attn_1_sub.append(attn1_.cpu().data.numpy())
				test_softmax_p2, attn2,head_output2 = get_softmax_map(model_WQ2,model_WK2, model_WV2,model_W02, attn1_)
				attn_2.append(attn2.cpu().data.numpy())
				head_output_2.append(head_output2.cpu().data.numpy())
				attn2 = attn2.view(-1, attn2.shape[1])

				attn2_ = model.sublayer(attn1_, attn2)
				attn_2_sub.append(attn2_.cpu().data.numpy())
				test_softmax_p3, attn3,head_output3 = get_softmax_map(model_WQ3,model_WK3, model_WV3,model_W03, attn2_)
				attn_3.append(attn3.cpu().data.numpy())
				head_output_3.append(head_output3.cpu().data.numpy())
				attn3 = attn3.view(-1, attn3.shape[1])

				attn3_ = model.sublayer(attn2_, attn3)
				attn_3_sub.append(attn3_.cpu().data.numpy())

				y_pred = model.fc(attn3_)
				pred = F.log_softmax(y_pred, dim=1).argmax(dim=1, keepdim=True)
				correct_test += pred.eq(batch_y_test.view_as(pred)).sum().item()

				layer_entropy1_h = []
				test_softmax_p_1_h_avg = []
				test_softmax_p_1_h_entropy = []
				layer_entropy2_h = []
				test_softmax_p_2_h_avg = []
				test_softmax_p_2_h_entropy = []
				test_softmax_p_3_h_entropy = []
				layer_entropy3_h = []
				test_softmax_p_3_h_avg = []
				for ind_h in range(n_head):
					layer_entropy1_h.append(scipy.stats.entropy(test_softmax_p1[ind_h].cpu().data.numpy(), axis=2))

					test_softmax_p_1_h_avg.append(np.sum(test_softmax_p1[ind_h].cpu().data.numpy(), axis=0))
					test_softmax_p_1_h_entropy.append(
						scipy.stats.entropy(test_softmax_p1[ind_h].cpu().data.numpy(), axis=2))

					test_softmax_p_2_h_entropy.append(
						scipy.stats.entropy(test_softmax_p2[ind_h].cpu().data.numpy(), axis=2))
					layer_entropy2_h.append(scipy.stats.entropy(test_softmax_p2[ind_h].cpu().data.numpy(), axis=2))

					test_softmax_p_2_h_avg.append(np.sum(test_softmax_p2[ind_h].cpu().data.numpy(), axis=0))
					test_softmax_p_3_h_entropy.append(
						scipy.stats.entropy(test_softmax_p3[ind_h].cpu().data.numpy(), axis=2))
					layer_entropy3_h.append(scipy.stats.entropy(test_softmax_p3[ind_h].cpu().data.numpy(), axis=2))

					test_softmax_p_3_h_avg.append(np.sum(test_softmax_p3[ind_h].cpu().data.numpy(), axis=0))

				layer_entropy1.append(layer_entropy1_h)
				layer_entropy2.append(layer_entropy2_h)
				layer_entropy3.append(layer_entropy3_h)
				test_softmax_p_1_avg.append(test_softmax_p_1_h_avg)
				test_softmax_p_2_avg.append(test_softmax_p_2_h_avg)
				test_softmax_p_3_avg.append(test_softmax_p_3_h_avg)
				test_softmax_p_1_entropy.append(test_softmax_p_1_h_entropy)
				test_softmax_p_2_entropy.append(test_softmax_p_2_h_entropy)
				test_softmax_p_3_entropy.append(test_softmax_p_3_h_entropy)

		test_layer_entropy1_all = np.concatenate(layer_entropy1, axis=1)
		test_layer_entropy2_all = np.concatenate(layer_entropy2, axis=1)
		test_layer_entropy3_all = np.concatenate(layer_entropy3, axis=1)
		test_softmax_p_1_avg_all = np.sum(test_softmax_p_1_avg, axis=0)
		test_softmax_p_2_avg_all = np.sum(test_softmax_p_2_avg, axis=0)
		test_softmax_p_3_avg_all = np.sum(test_softmax_p_3_avg, axis=0)
		test_softmax_p_1_entropy_all = np.concatenate(test_softmax_p_1_entropy, axis=1)
		test_softmax_p_2_entropy_all = np.concatenate(test_softmax_p_2_entropy, axis=1)
		test_softmax_p_3_entropy_all = np.concatenate(test_softmax_p_3_entropy, axis=1)
		test_softmax_p_ave = np.stack((test_softmax_p_1_avg_all / n_samples, test_softmax_p_2_avg_all / n_samples,
										test_softmax_p_3_avg_all / n_samples))

		print(correct_test / len(X_test_input))
		attn_1_test=np.vstack(attn_1)
		attn_2_test=np.vstack(attn_2)
		attn_3_test=np.vstack(attn_3)
		attn_1_sub_test=np.vstack(attn_1_sub)
		attn_2_sub_test=np.vstack(attn_2_sub)
		attn_3_sub_test=np.vstack(attn_3_sub)
		head_output_1_test = np.vstack(head_output_1)
		head_output_2_test = np.vstack(head_output_2)
		head_output_3_test=np.vstack(head_output_3)

		y_test_attn=np.hstack(y_test_attn)

		attn={}
		head_output={}
		softmax_entropy={}

		softmax_entropy['test_layer_entropy1_all']=test_layer_entropy1_all
		softmax_entropy['test_layer_entropy2_all']=test_layer_entropy2_all
		softmax_entropy['test_layer_entropy3_all']=test_layer_entropy3_all

		softmax_entropy['test_softmax_p_1_avg_all']=test_softmax_p_1_avg_all
		softmax_entropy['test_softmax_p_2_avg_all']=test_softmax_p_2_avg_all
		softmax_entropy['test_softmax_p_3_avg_all']=test_softmax_p_3_avg_all
		softmax_entropy['test_softmax_p_1_entropy_all']=test_softmax_p_1_entropy_all
		softmax_entropy['test_softmax_p_2_entropy_all']=test_softmax_p_2_entropy_all
		softmax_entropy['test_softmax_p_3_entropy_all']=test_softmax_p_3_entropy_all
		softmax_entropy['test_softmax_p_ave']=test_softmax_p_ave

		softmax_entropy['train_layer_entropy1_all']=train_layer_entropy1_all
		softmax_entropy['train_layer_entropy2_all']=train_layer_entropy2_all
		softmax_entropy['train_layer_entropy3_all']=train_layer_entropy3_all

		softmax_entropy['train_softmax_p_1_avg_all']=train_softmax_p_1_avg_all
		softmax_entropy['train_softmax_p_2_avg_all']=train_softmax_p_2_avg_all
		softmax_entropy['train_softmax_p_3_avg_all']=train_softmax_p_3_avg_all
		softmax_entropy['train_softmax_p_1_entropy_all']=train_softmax_p_1_entropy_all
		softmax_entropy['train_softmax_p_2_entropy_all']=train_softmax_p_2_entropy_all
		softmax_entropy['train_softmax_p_3_entropy_all']=train_softmax_p_3_entropy_all
		softmax_entropy['train_softmax_p_ave']=train_softmax_p_ave

		attn['y_train_attn']=y_train_attn
		attn['attn_1_test']=attn_1_test
		attn['attn_2_test']=attn_2_test
		attn['attn_3_test']=attn_3_test

		attn['attn_1_sub_test']=attn_1_sub_test
		attn['attn_2_sub_test']=attn_2_sub_test
		attn['attn_3_sub_test']=attn_3_sub_test

		attn['y_test_attn']=y_test_attn
		attn['attn_1_train']=attn_1_train
		attn['attn_2_train']=attn_2_train
		attn['attn_3_train']=attn_3_train

		attn['attn_1_sub_train']=attn_1_sub_train
		attn['attn_2_sub_train']=attn_2_sub_train
		attn['attn_3_sub_train']=attn_3_sub_train


		head_output['head_output_1_train']=head_output_1_train
		head_output['head_output_2_train']=head_output_2_train
		head_output['head_output_3_train']=head_output_3_train

		head_output['head_output_1_test']=head_output_1_test
		head_output['head_output_2_test'] = head_output_2_test
		head_output['head_output_3_test'] = head_output_3_test


		pl.dump(attn,open("./model_res_vis_all/attn_3l_product.plk","wb"))
		pl.dump(head_output,open("./model_res_vis_all/head_output_3l_product.plk","wb"))
		pl.dump(softmax_entropy,open("./model_res_vis_all/softmax_entropy_3l_product.plk","wb"))


	################ heatmap for all samples (training)
	if fun=='vis':
		attn_dict=pl.load(open("./model_res_vis_all/attn_3l_product.plk","rb"))
		attn_1_test=attn_dict['attn_1_test']
		attn_2_test=attn_dict['attn_2_test']
		attn_3_test=attn_dict['attn_3_test']
		y_test_attn=attn_dict['y_test_attn']

		attn_1_train=attn_dict['attn_1_train']
		attn_2_train=attn_dict['attn_2_train']
		attn_3_train=attn_dict['attn_3_train']
		y_train_attn=attn_dict['y_train_attn']

		attn_1_sub_train=attn_dict['attn_1_sub_train']
		attn_2_sub_train=attn_dict['attn_2_sub_train']
		attn_3_sub_train=attn_dict['attn_3_sub_train']

		head_output_dict=pl.load(open("./model_res_vis_all/head_output_3l_product.plk","rb"))
		head_output_1_test=head_output_dict['head_output_1_test']
		head_output_2_test=head_output_dict['head_output_2_test']
		head_output_3_test=head_output_dict['head_output_3_test']

		head_output_1_train=head_output_dict['head_output_1_train']
		head_output_2_train=head_output_dict['head_output_2_train']
		head_output_3_train=head_output_dict['head_output_3_train']

		softmax_entropy=pl.load(open("./model_res_vis_all/softmax_entropy_3l_product.plk","rb"))

		test_layer_entropy1_all=softmax_entropy['test_layer_entropy1_all']
		test_layer_entropy2_all=softmax_entropy['test_layer_entropy2_all']
		test_layer_entropy3_all=softmax_entropy['test_layer_entropy3_all']
		test_softmax_p_1_avg_all=softmax_entropy['test_softmax_p_1_avg_all']
		test_softmax_p_2_avg_all=softmax_entropy['test_softmax_p_2_avg_all']
		test_softmax_p_3_avg_all=softmax_entropy['test_softmax_p_3_avg_all']
		test_softmax_p_1_entropy_all=softmax_entropy['test_softmax_p_1_entropy_all']
		test_softmax_p_2_entropy_all=softmax_entropy['test_softmax_p_2_entropy_all']
		test_softmax_p_3_entropy_all=softmax_entropy['test_softmax_p_3_entropy_all']
		test_softmax_p_ave=softmax_entropy['test_softmax_p_ave']

		train_layer_entropy1_all=softmax_entropy['train_layer_entropy1_all']
		train_layer_entropy2_all=softmax_entropy['train_layer_entropy2_all']
		train_layer_entropy3_all=softmax_entropy['train_layer_entropy3_all']
		train_softmax_p_1_avg_all=softmax_entropy['train_softmax_p_1_avg_all']
		train_softmax_p_2_avg_all=softmax_entropy['train_softmax_p_2_avg_all']
		train_softmax_p_3_avg_all=softmax_entropy['train_softmax_p_3_avg_all']
		train_softmax_p_1_entropy_all=softmax_entropy['train_softmax_p_1_entropy_all']
		train_softmax_p_2_entropy_all=softmax_entropy['train_softmax_p_2_entropy_all']
		train_softmax_p_3_entropy_all=softmax_entropy['train_softmax_p_3_entropy_all']
		train_softmax_p_ave=softmax_entropy['train_softmax_p_ave']

		head_entropy = []
		for head in range(5):
			head_entropy.append(np.mean(test_layer_entropy1_all[head, ::]))
			head_entropy.append(np.mean(test_layer_entropy2_all[head, ::]))
			head_entropy.append(np.mean(test_layer_entropy3_all[head, ::]))

		np.savetxt('model_res_vis_all/mean_entropy.txt', np.array(head_entropy).reshape(5,3))


		###################### boxplot for each layer's entropy
		seaborn.boxplot(data=np.mean(test_layer_entropy1_all, axis=1).T)
		seaborn.boxplot(data=np.mean(test_layer_entropy2_all, axis=1).T)
		seaborn.boxplot(data=np.mean(test_layer_entropy3_all, axis=1).T)
		seaborn.boxplot(data=np.hstack((np.mean(test_layer_entropy1_all, axis=1).T, np.mean(test_layer_entropy2_all, axis=1).T,
							np.mean(test_layer_entropy3_all, axis=1).T)))


		###get the average head out put and box plot
		seaborn.boxplot(data=np.mean(head_output_1_test, axis=0))
		seaborn.boxplot(data=np.mean(head_output_2_test, axis=0))
		seaborn.boxplot(data=np.mean(head_output_3_test, axis=0))
		seaborn.boxplot(data=np.hstack((np.mean(head_output_1_test, axis=0), np.mean(head_output_2_test, axis=0),
							np.mean(head_output_3_test, axis=0))))
		############boxplot
		seaborn.boxplot(data=np.vstack((np.mean(attn_1_test, axis=0), np.mean(attn_2_test, axis=0),
										np.mean(attn_3_test, axis=0))))


		#
		# #################check the top 20 genes from entropy and head output
		# # head=0
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy1_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top20 = np.array(gene_list)[sorted_index_entropy][0:20]
		# 	sorted_index_headout = np.argsort(-np.mean(head_output_1_test[:, :, head], axis=0))
		# 	sorted_gene_headout_top20 = np.array(gene_list)[sorted_index_headout][0:20]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_1)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top20 = np.array(gene_list)[sorted_index_V][0:20]
		# 	f = open('model_res_vis_all/sorted gene via entropy and head output and V layer 1.txt', 'a')
		# 	f.writelines('top 20 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top20))
		# 	f.write('\n')
		# 	f.writelines('top 20 gene with most head output head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top20))
		# 	f.write('\n')
		#
		# 	f.close()
		#
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy2_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top20 = np.array(gene_list)[sorted_index_entropy][0:20]
		# 	sorted_index_headout = np.argsort(-np.mean(head_output_2_test[:, :, head], axis=0))
		# 	sorted_gene_headout_top20 = np.array(gene_list)[sorted_index_headout][0:20]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_2)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top20 = np.array(gene_list)[sorted_index_V][0:20]
		# 	f = open('model_res_vis_all/sorted gene via entropy and head output and V layer 2.txt', 'a')
		# 	f.writelines('top 20 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top20))
		# 	f.write('\n')
		# 	f.writelines('top 20 gene with most head output head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top20))
		# 	f.write('\n')
		# 	# f.writelines('top 20 gene with most V head ' + str(head))
		# 	# f.write('\n')
		# 	# f.writelines(str(sorted_gene_V_top20))
		# 	# f.write('\n')
		# 	f.close()
		#
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy3_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top20 = np.array(gene_list)[sorted_index_entropy][0:20]
		# 	sorted_index_headout = np.argsort(-np.mean(head_output_3_test[:, :, head], axis=0))
		# 	sorted_gene_headout_top20 = np.array(gene_list)[sorted_index_headout][0:20]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_3)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top20 = np.array(gene_list)[sorted_index_V][0:20]
		# 	f = open('model_res_vis_all/sorted gene via entropy and head and V output layer 3.txt', 'a')
		# 	f.writelines('top 20 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top20))
		# 	f.write('\n')
		# 	f.writelines('top 20 gene with most head output head and V ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top20))
		# 	f.write('\n')
		# 	# f.writelines('top 20 gene with most V head ' + str(head))
		# 	# f.write('\n')
		# 	# f.writelines(str(sorted_gene_V_top20))
		# 	# f.write('\n')
		# 	f.close()

		#
		# ###get the average and abs head out put and box plot
		# seaborn.boxplot(data=np.abs(np.mean(head_output_1_test, axis=0)))
		# seaborn.boxplot(data=np.abs(np.mean(head_output_2_test, axis=0)))
		# seaborn.boxplot(data=np.abs(np.mean(head_output_3_test, axis=0)))

		#
		# #################check the top 10 genes from entropy and head output
		# # head=0
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy1_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top10 = np.array(gene_list)[sorted_index_entropy][0:10]
		# 	sorted_index_headout = np.argsort(-np.abs(np.abs(np.mean(head_output_1_test[:, :, head], axis=0))))
		# 	sorted_gene_headout_top10 = np.array(gene_list)[sorted_index_headout][0:10]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_1)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top10 = np.array(gene_list)[sorted_index_V][0:10]
		# 	f = open('model_res_vis_all/sorted gene via entropy and head output abs layer 1.txt', 'a')
		# 	f.writelines('top 10 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top10))
		# 	f.write('\n')
		# 	f.writelines('top 10 gene with most head output head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top10))
		# 	f.write('\n')
		# 	# f.writelines('top 10 gene with most V head ' + str(head))
		# 	# f.write('\n')
		# 	# f.writelines(str(sorted_gene_V_top10))
		# 	# f.write('\n')
		# 	f.close()
		#
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy2_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top10 = np.array(gene_list)[sorted_index_entropy][0:10]
		# 	sorted_index_headout = np.argsort(-np.abs(np.mean(head_output_2_test[:, :, head], axis=0)))
		# 	sorted_gene_headout_top10 = np.array(gene_list)[sorted_index_headout][0:10]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_2)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top10 = np.array(gene_list)[sorted_index_V][0:10]
		# 	f = open('model_res_vis_all/sorted gene via entropy and head output abs layer 2.txt', 'a')
		# 	f.writelines('top 10 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top10))
		# 	f.write('\n')
		# 	f.writelines('top 10 gene with most head output head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top10))
		# 	f.write('\n')
		# 	# f.writelines('top 10 gene with most V head ' + str(head))
		# 	# f.write('\n')
		# 	# f.writelines(str(sorted_gene_V_top10))
		# 	# f.write('\n')
		# 	f.close()
		#
		# for head in range(5):
		# 	sorted_index_entropy = np.argsort(np.mean(test_layer_entropy3_all[head, :, :], axis=0))
		# 	sorted_gene_entropy_top10 = np.array(gene_list)[sorted_index_entropy][0:10]
		# 	sorted_index_headout = np.argsort(-np.abs(np.mean(head_output_3_test[:, :, head], axis=0)))
		# 	sorted_gene_headout_top10 = np.array(gene_list)[sorted_index_headout][0:10]
		# 	# sorted_index_V = np.argsort(-np.mean(np.array(V_seq_3)[head,:, :,], axis=0))
		# 	# sorted_gene_V_top10 = np.array(gene_list)[sorted_index_V][0:10]
		# 	f = open('model_res_vis_all/sorted gene via entropy and  head output abs layer 3.txt', 'a')
		# 	f.writelines('top 10 gene with least entropy head ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_entropy_top10))
		# 	f.write('\n')
		# 	f.writelines('top 10 gene with most head output head and V ' + str(head))
		# 	f.write('\n')
		# 	f.writelines(str(sorted_gene_headout_top10))
		# 	f.write('\n')
		# 	# f.writelines('top 10 gene with most V head ' + str(head))
		# 	# f.write('\n')
		# 	# f.writelines(str(sorted_gene_V_top10))
		# 	# f.write('\n')
		# 	f.close()

