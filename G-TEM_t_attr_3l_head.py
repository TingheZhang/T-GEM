import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import csv
import scipy
import numpy as np
import pickle as plk
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
import pickle as pl

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os, copy, sys
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools
# from torch.utils.data.dataset import Dataset
# from torchvision import transforms
# from gpu_mem_track import MemTracker
import inspect
# import torch.utils.checkpoint as cp
from torch.utils.checkpoint import checkpoint as cp
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
	"--do_val", action="store_true", help="Whether split the val set from train set"
)
parser.add_argument(
	"--attr_train", action="store_true", help="Whether compute attribution on train set"
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


args = parser.parse_args()

##################
print(sys.argv)
d_ff = 1024
dropout_rate = args.dropout_rate
n_epochs = args.epoch
batch_size = args.batch_size
n_layers=3
n_head = args.head_num
lr_rate = args.learning_rate
act_fun = args.act_fun
gain = 1
include_train=args.attr_train
rand_state = args.rand_seed
n_gene = 1708
n_feature = 1708
# n_class=0
n_class = 34
query_gene = 64  # not using but cannot delete
val = args.do_val

save_memory = False
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class mulitiattention_out(torch.nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, mode, V, W0):
        """
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
        # gpu_tracker.track()
        super(mulitiattention_out, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.mode = mode
        self.query_gene = query_gene
        self.WV = V
        self.W_0 = W0
        # self.WQ = nn.Parameter(torch.Tensor(self.n_head, n_feature, 1), requires_grad=True)
        # self.WK = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)
        # self.WV = nn.Parameter(torch.Tensor(self.n_head,n_feature,1),requires_grad=True)
        # # torch.nn.init.xavier_uniform_(self.WK,gain=50)
        # # torch.nn.init.xavier_uniform_(self.WV,gain=50)
        # # torch.nn.init.xavier_normal_(self.WK,gain=25)
        # # torch.nn.init.xavier_normal_(self.WV,gain=25)
        # torch.nn.init.xavier_normal_(self.WQ,gain=gain)
        # torch.nn.init.xavier_normal_(self.WK,gain=gain)
        # # torch.nn.init.xavier_normal_(self.WQ,gain=0)
        # # torch.nn.init.xavier_normal_(self.WK,gain=0)
        # torch.nn.init.xavier_normal_(self.WV)
        # seaborn.violinplot(x=self.WK.cpu().data.numpy())
        # seaborn.violinplot(x=self.WV.cpu().data.numpy())
        self.W_0 = W0

        print('init')

    # gpu_tracker.track()

    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)
        # QK_dif = -1 * (Q_seq - K_seq) * (Q_seq - K_seq)
        # QK_dif = -1 * ((Q_seq - K_seq)**2)
        # QK_dif.cpu().data.numpy()
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self, x):
        # for d in range(x.shape[1]):
        # 	x[:,d,d]=0
        # n=x.shape[0]
        d = x.shape[1]
        x = x * ((1 - torch.eye(d, d)).to(device))
        return x

    def attention_out(self, x, WV, input):
        V_seq = input * WV
        z = self.mask_softmax_self(x)

        # z=self.dropout(z)
        out_seq = torch.matmul(z, V_seq)

        return out_seq

    def forward(self, x, input):

        # gpu_tracker.track()
        input = torch.reshape(input, (input.shape[0], input.shape[1], 1))
        # x = x.expand(x.shape[0], x.shape[1], self.n_head)
        # Q_seq=x*self.WQ
        # Q_seq=Q_seq.expand(Q_seq.shape[0],Q_seq.shape[1],self.n_gene)
        out_h = []
        for h in range(self.n_head):
            if save_memory:
                attn_out = cp(self.attention_out, x[:, h, :], self.WV[h, :, :], input)
            else:
                attn_out = self.attention_out(x[:, h, :], self.WV[h, :, :], input)

            out_h.append(attn_out)
        out_seq = torch.cat(out_h, dim=2)
        out_seq = torch.matmul(out_seq, self.W_0)
        return out_seq


class mulitiattention(torch.nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, query_gene, mode):
        """
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
        # gpu_tracker.track()
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
        # torch.nn.init.xavier_uniform_(self.WK,gain=50)
        # torch.nn.init.xavier_uniform_(self.WV,gain=50)
        # torch.nn.init.xavier_normal_(self.WK,gain=25)
        # torch.nn.init.xavier_normal_(self.WV,gain=25)
        torch.nn.init.xavier_normal_(self.WQ, gain=gain)
        torch.nn.init.xavier_normal_(self.WK, gain=gain)
        # torch.nn.init.xavier_normal_(self.WQ,gain=0)
        # torch.nn.init.xavier_normal_(self.WK,gain=0)
        torch.nn.init.xavier_normal_(self.WV)
        # seaborn.violinplot(x=self.WK.cpu().data.numpy())
        # seaborn.violinplot(x=self.WV.cpu().data.numpy())
        self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]), requires_grad=True)

        print('init')

    # gpu_tracker.track()

    def QK_diff(self, Q_seq, K_seq):
        QK_dif = -1 * torch.pow((Q_seq - K_seq), 2)
        # QK_dif = -1 * (Q_seq - K_seq) * (Q_seq - K_seq)
        # QK_dif = -1 * ((Q_seq - K_seq)**2)
        # QK_dif.cpu().data.numpy()
        return torch.nn.Softmax(dim=2)(QK_dif)

    def mask_softmax_self(self, x):
        # for d in range(x.shape[1]):
        # 	x[:,d,d]=0
        # n=x.shape[0]
        d = x.shape[1]
        x = x * ((1 - torch.eye(d, d)).to(device))
        return x

    def attention(self, x, Q_seq, WK, WV):
        if self.mode == 0:
            K_seq = x * WK
            K_seq = K_seq.expand(K_seq.shape[0], K_seq.shape[1], self.n_gene)
            K_seq = K_seq.permute(0, 2, 1)
            V_seq = x * WV
            # print(Q_seq.detach().numpy())
            # print(x.detach().numpy())
            # print(self.WK.detach().numpy())
            # print(self.WV.detach().numpy())
            # print(K_seq.detach().numpy())
            # print(V_seq.detach().numpy())
            # QK_diff=-1*(Q_seq-K_seq)*(Q_seq-K_seq)
            # z=torch.nn.Softmax(dim=2)(QK_diff)
            # z = self.QK_diff(Q_seq, K_seq)
            QK_product = Q_seq * K_seq
            out_softmax = torch.nn.Softmax(dim=2)(QK_product)

        return out_softmax

    def forward(self, x):

        # gpu_tracker.track()
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        # x = x.expand(x.shape[0], x.shape[1], self.n_head)
        # Q_seq=x*self.WQ
        # Q_seq=Q_seq.expand(Q_seq.shape[0],Q_seq.shape[1],self.n_gene)
        out_h = []
        for h in range(self.n_head):
            Q_seq = x * self.WQ[h, :, :]
            Q_seq = Q_seq.expand(Q_seq.shape[0], Q_seq.shape[1], self.n_gene)
            if save_memory:
                attention_out = cp(self.attention, x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])
            else:
                attention_out = self.attention(x, Q_seq, self.WK[h, :, :], self.WV[h, :, :])

            out_h.append(attention_out)
        out_softmax = torch.stack(out_h, dim=1)
        # out_seq=torch.cat(out_h,dim=2)
        # out_seq=torch.matmul(out_seq,self.W_0)
        return out_softmax


class layernorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

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
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(res_connect, self).__init__()
        self.norm = layernorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        "Apply residual connection to any sublayer with the same size."
        return x + self.norm(self.dropout(out))


class MyNet(torch.nn.Module):
    def __init__(self, batch_size, n_head, n_gene, n_feature, n_class, query_gene, d_ff, dropout_rate, mode, model_raw):
        super(MyNet, self).__init__()
        self.n_head = n_head
        self.n_gene = n_gene
        self.batch_size = batch_size
        self.n_feature = n_feature
        self.n_class = n_class
        self.d_ff = d_ff
        # self.wpool = weight_pool((1, 2))
        self.mulitiattention1 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
                                                mode)
        self.mulitiattention1_out = mulitiattention_out(self.batch_size, self.n_head, self.n_gene, self.n_feature,
                                                        query_gene,
                                                        mode, model_raw.mulitiattention1.WV,
                                                        model_raw.mulitiattention1.W_0)

        self.mulitiattention2 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
                                                mode)
        self.mulitiattention2_out = mulitiattention_out(self.batch_size, self.n_head, self.n_gene, self.n_feature,
                                                        query_gene,
                                                        mode, model_raw.mulitiattention2.WV,
                                                        model_raw.mulitiattention2.W_0)

        self.mulitiattention3 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
                                                mode)
        self.mulitiattention3_out = mulitiattention_out(self.batch_size, self.n_head, self.n_gene, self.n_feature,
                                                        query_gene,
                                                        mode, model_raw.mulitiattention3.WV,
                                                        model_raw.mulitiattention3.W_0)
        # self.mulitiattention4 = mulitiattention(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
        #                                        mode)
        # self.mulitiattention4_out = mulitiattention_out(self.batch_size, self.n_head, self.n_gene, self.n_feature, query_gene,
        # 										mode,model_raw.mulitiattention4.WV,model_raw.mulitiattention4.W_0)

        # self.fc=torch.nn.Linear(self.n_gene,1)
        self.fc = nn.Linear(self.n_gene, self.n_class)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)
        self.ffn1 = nn.Linear(self.n_gene, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff, self.n_gene)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = res_connect(n_gene, dropout_rate)
        self.norm = layernorm(n_gene)

    # self.dropout = nn.Dropout(dropout_rate)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        # x=x.view(x.shape[0],x.shape[1],1)
        # out=F.relu(nn.Conv1d(x.shape[1],self.d_ff,1).to(device)(x))
        # out = nn.Conv1d(self.d_ff,x.shape[1],1).to(device)(out)
        # out=out.view(x.shape[0],x.shape[1])
        return out

    def forward(self, x):
        # gpu_tracker.track()
        out_softmax = self.mulitiattention1(x)
        out_attn = self.mulitiattention1_out(out_softmax, x)
        # out_attn = self.norm(out_attn)
        out_attn_1 = self.sublayer(x, out_attn)
        out_softmax_2 = self.mulitiattention2(out_attn_1)
        out_attn_2 = self.mulitiattention2_out(out_softmax_2, out_attn_1)

        out_attn_2 = self.sublayer(out_attn_1, out_attn_2)
        out_softmax_3 = self.mulitiattention3(out_attn_2)
        out_attn_3 = self.mulitiattention3_out(out_softmax_3, out_attn_2)
        out_attn_3 = self.sublayer(out_attn_2, out_attn_3)
        # out_softmax_4 = self.mulitiattention4(out_attn_3)
        # out_attn_4 = self.mulitiattention4_out(out_softmax_4, out_attn_3)
        # out_attn_4=self.sublayer(out_attn_3,out_attn_4)
        # out_attn_=F.relu(out_attn_3)
        # out_attn= self.mulitiattention(out_attn)
        # out_attn=F.relu(out_attn)
        if act_fun == 'relu':
            out_attn_3 = F.relu(out_attn_3)
        if act_fun == 'leakyrelu':
            m = torch.nn.LeakyReLU(0.1)
            out_attn_3 = m(out_attn_3)
        if act_fun == 'gelu':
            m = torch.nn.GELU()
            out_attn_3 = m(out_attn_3)
        y_pred = self.fc(out_attn_3)

        # y_pred=F.softmax(y_pred,dim=1)
        y_pred = F.log_softmax(y_pred, dim=1)
        # gpu_tracker.track()
        # y_pred = F.relu(y_pred)
        # print(y_pred.detach().numpy())
        return y_pred


def get_percent_acc(attr_sort_index, xx, yy, percent_to_be_0):
    acc = []
    # X_train_input[:,attr_sort_index[0:int(len(attr_sort_index)*percent)]]
    # X_test_input_percent=torch.from_numpy(np.vstack(X_test)[:, attr_sort_index[0:int(len(attr_sort_index) * percent)]])
    # X_test_input_percent = np.vstack(X_test)
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
            # batch_x_test, batch_y_test = X_test_input_percent_tensor[indices], y_test_input[indices]
            batch_x_test, batch_y_test = X_test_input_percent_tensor[indices], torch.from_numpy(yy)[
                indices]
            batch_x_test, batch_y_test = batch_x_test.to(device), batch_y_test.to(device)
            output_test = model(batch_x_test.float())

            # X_var = Variable(batch_x_test.float(), requires_grad=True)
            # y_pred = model(X_var)
            # y_pred = y_pred.gather(1, batch_y_test.view(-1, 1)).squeeze()
            # y_pred.backward(torch.FloatTensor([1.0] * batch_y_test.shape[0]).to(device))

            test_loss += F.nll_loss(output_test, batch_y_test, reduction='sum')
            # test_loss=criterion(output_test, batch_y_test.float())
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output_test.argmax(dim=1, keepdim=True)
            # batch_y_class=batch_y_test.argmax(dim=1, keepdim=True)# get the index of the max log-probability
            # correct += pred.eq(batch_y_class.view_as(pred)).sum().item()
            correct += pred.eq(batch_y_test.view_as(pred)).sum().item()

    test_loss /= len(X_test_input)
    acc.append(correct / len(X_test_input_percent_tensor))
    return acc


if __name__ == '__main__':

    y, data_df, pathway_gene, pathway, cancer_name = pl.load(open('./pathway_data.pckl', 'rb'))
    data_ = np.array(data_df)
    x = np.float32(data_)
    gene_list = data_df.columns.tolist()

    x = np.float32(data_)
    # x = scipy.stats.zscore(x, axis=None)
    encoder = LabelEncoder()
    y_label = encoder.fit_transform(y)
    class_label = np.unique(y)

    u, count = np.unique(y_label, return_counts=True)
    count_sort_ind = np.argsort(-count)
    y_label_unique_top34 = u[count_sort_ind[0:34]]
    # class_label[y_label_unique_top10]

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
    # X_train, X_test, y_train, y_test = train_test_split(np.vstack(x_top10), np.hstack(y_top10), test_size=0.2, random_state=52)
    # X_train, X_test, y_train, y_test = train_test_split(x, y_label, test_size=0.2, random_state=52)

    if val == True:
        # X_train, X_val, y_train, y_val = train_test_split(np.vstack(X_train), np.hstack(y_train), test_size=0.1, random_state=52)
        # X_val_input=torch.from_numpy(X_val)
        # y_val_input=torch.from_numpy(y_val)
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


    model_addLayer = MyNet(batch_size, n_head, n_gene, n_feature, n_class, query_gene, d_ff, dropout_rate, mode=0,
                           model_raw=model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0, amsgrad=False)  ## val acc 98.75
    # model1.mulitiattention1_out
    model.mulitiattention1_out = model_addLayer.mulitiattention1_out
    model.mulitiattention2_out = model_addLayer.mulitiattention2_out
    model.mulitiattention3_out = model_addLayer.mulitiattention3_out
    # model.mulitiattention4_out=model_addLayer.mulitiattention4_out

    train_loss_list = []
    val_loss_list = []
    res = {}
    confusion_matrix_res = []
    mcc_res = []
    acc_res = []
    auc_res = []
    f1_res = []

    # get the attr for all 0 cancers here
    attr_ig_ave_acc_mean_list = []
    attr_ig_med_acc_mean_list = []
    attr_ig_ave_acc_mean_list_10 = []
    attr_ig_med_acc_mean_list_10 = []
    attr_ig_ave_acc_mean_score_list = []
    attr_ig_med_acc_mean_score_list = []
    attr_ig_ave_acc_mean_score_list_10 = []
    attr_ig_med_acc_mean_score_list_10 = []

    # try
    # IG
    cancer_type = 0
    # attr_method == 'ig'
    from captum.attr import LayerConductance, LayerIntegratedGradients

    torch.manual_seed(123)
    np.random.seed(123)
    model.eval()
    model = model.to(device)

    attributions_list_mean_all = []
    for layer in range(n_layers):
        # lc=LayerConductance(model,model.mulitiattention3)
        lig = eval(
            'LayerIntegratedGradients(model,model.mulitiattention' + str(layer + 1) + ',multiply_by_inputs=True)')
        permutation = torch.randperm(torch.from_numpy(X_test[cancer_type]).size()[0])
        # torch.cuda.empty_cache()
        n_correct, n_total = 0, 0
        attributions_list = []
        for batch_idx, i in enumerate(tqdm(range(0, torch.from_numpy(X_test[cancer_type]).size()[0], 1))):
            indices = permutation[i:i + 1]
            batch_x, batch_y = torch.from_numpy(X_test[cancer_type])[indices], \
                               torch.from_numpy(y_test[cancer_type])[indices]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            attributions, approximation_error = lig.attribute(batch_x, target=batch_y,
                                                              return_convergence_delta=True, baselines=0)

            attributions_list.append(attributions.detach().cpu().numpy())
        ####### if I put two layers there the attributions will become two list of attribtuions
        attributions_list_mean = np.mean(attributions_list, axis=0)
        attributions_list_mean_all.append(attributions_list_mean)
    plk.dump(attributions_list_mean_all, open(args.result_dir+'attr_softmax_alllayer.plk', 'wb'))
