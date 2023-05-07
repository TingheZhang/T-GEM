import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

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

#### comment this if you are not using GPU
torch.set_num_threads(10)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##########
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
    "--do_val", action="store_true", help="Whether do validation or not"
)
parser.add_argument(
    "--result_dir",
    required=True,
    type=str,
    help="The dir used to save result and loss figure",
)
parser.add_argument(
    "--model_dir",
    required=True,
    type=str,
    help="The dir used to save model for each epoch ",
)
args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir, exist_ok=True)
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir, exist_ok=True)
if not os.path.exists(args.result_dir + '/model_figure'):
    os.makedirs(args.result_dir + '/model_figure/', exist_ok=True)

# print(sys.argv)
d_ff = 1024
dropout_rate = args.dropout_rate
n_epochs = args.epoch
batch_size = args.batch_size
# n_head = np.int(sys.argv[1])
# lr_rate=np.double(sys.argv[2])
n_head = args.head_num
lr_rate = args.learning_rate
# rand_state=np.int(sys.argv[4])
act_fun = args.act_fun
gain = 1

rand_state = args.rand_seed
n_gene = 1000
n_feature = 1000
# n_class=0
n_class = 10
query_gene = 64
val = args.do_val

# save_memory=True
save_memory = False
# gpu_tracker.track()
# model = attention(batch_size,n_head,n_gene,n_feature)
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
        torch.nn.init.xavier_normal_(self.WQ, gain=1)
        torch.nn.init.xavier_normal_(self.WK, gain=1)

        torch.nn.init.xavier_normal_(self.WV)
        self.W_0 = nn.Parameter(torch.Tensor(self.n_head * [0.001]), requires_grad=True)
        print('init')
        # gpu_tracker.track()

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

        ############this part is not working well
        if self.mode == 1:
            zz_list = []
            for q in range(self.n_gene // self.query_gene):
                # gpu_tracker.track()
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
            ####################################
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

    ##########    A residual connection followed by a layer norm.

    def __init__(self, size, dropout):
        super(res_connect, self).__init__()
        self.norm = layernorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, out):
        ###Apply residual connection to any sublayer with the same size
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
        self.fc = nn.Linear(self.n_gene, self.n_class)
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=1)
        self.ffn1 = nn.Linear(self.n_gene, self.d_ff)
        self.ffn2 = nn.Linear(self.d_ff, self.n_gene)
        self.dropout = nn.Dropout(dropout_rate)
        self.sublayer = res_connect(n_gene, dropout_rate)

    def feedforward(self, x):
        out = F.relu(self.ffn1(x))
        out = self.ffn2(self.dropout(out))
        return out

    def forward(self, x):

        out_attn = self.mulitiattention1(x)
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


if __name__ == '__main__':

    x = pl.load(open('data_all_1000.plk', 'rb'))
    x = np.float32(x)
    y_label = pl.load(open('labels_number_1000.plk', 'rb'))

    encoder_ = LabelBinarizer()
    y_label_ = encoder_.fit_transform(y_label)
    #
    u, count = np.unique(y_label, return_counts=True)
    count_sort_ind = np.argsort(-count)
    y_label_unique_all10 = u[count_sort_ind[0:10]]
    x_all10 = []
    y_all10 = []
    sample_size = []

    for j, sample_label in enumerate(y_label_unique_all10):
        sample_index = np.argwhere(y_label == sample_label)[:, 0]
        sample_size.append(sample_index.shape)
        x_all10.append(x[sample_index])
        temp_y = y_label[sample_index]
        temp_y[temp_y == sample_label] = j
        y_all10.append(temp_y.astype('int'))

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for d in range(len(x_all10)):
        x_train, x_test, y_tr, y_te = train_test_split(x_all10[d], y_all10[d], test_size=0.2,
                                                       random_state=rand_state)
        X_train.append(x_train)
        X_test.append(x_test)
        y_train.append(y_tr)
        y_test.append(y_te)
    # X_train, X_test, y_train, y_test = train_test_split(np.vstack(x_all10), np.hstack(y_all10), test_size=0.2, random_state=52)
    # X_train, X_test, y_train, y_test = train_test_split(x, y_label, test_size=0.2, random_state=52)

    if val == True:
        # X_train, X_val, y_train, y_val = train_test_split(np.vstack(X_train), np.hstack(y_train), test_size=0.1, random_state=52)
        # X_val_input=torch.from_numpy(X_val)
        # y_val_input=torch.from_numpy(y_val)
        X_train_val = []
        X_val = []
        y_train_val = []
        y_val = []

        for dd in range(len(x_all10)):
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

    model = MyNet(batch_size, n_head, n_gene, n_feature, n_class, query_gene, d_ff, dropout_rate, mode=0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)  ## val acc 98.75
    # device = torch.device( "cpu")

    # device = torch.device("cuda")
    # para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # type_size=4
    # print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    train_loss_list = []
    val_loss_list = []
    res = {}
    confusion_matrix_res = []
    mcc_res = []
    acc_res = []
    auc_res = []
    f1_res = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        permutation = torch.randperm(X_train_input.size()[0])
        # torch.cuda.empty_cache()
        n_correct, n_total = 0, 0
        for batch_idx, i in enumerate(range(0, X_train_input.size()[0], batch_size)):
            model.train()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train_input[indices], y_train_input[indices]
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # Forward pass: Compute predicted y by passing x to the model
            optimizer.zero_grad()
            # gpu_tracker.track()
            y_pred = model(batch_x.float())

            loss = F.nll_loss(y_pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(batch_x), len(X_train_input),
                           100. * i / len(X_train_input), loss.item()))

        train_loss /= len(X_train_input)
        train_loss_list.append(train_loss)

        if val == True:
            permutation_val = torch.randperm(X_val_input.size()[0])

            correct_val = 0
            val_loss = 0

            with torch.no_grad():
                model.eval()
                batch_pred = []
                batch_y_val_list = []
                batch_pred_cate = []
                for batch_idx_val, i in enumerate(range(0, X_val_input.size()[0], batch_size)):

                    indices_val = permutation_val[i:i + batch_size]
                    batch_x_val, batch_y_val = X_val_input[indices_val], y_val_input[indices_val]
                    # batch_x_val, batch_y_val = X_test_input[indices_val], y_test_input[indices_val]
                    batch_x_val, batch_y_val = batch_x_val.to(device), batch_y_val.to(device)

                    output_val = model(batch_x_val.float())
                    val_loss += F.nll_loss(output_val, batch_y_val, reduction='sum')
                    pred_val = output_val.argmax(dim=1, keepdim=True)

                    correct_val += pred_val.eq(batch_y_val.view_as(pred_val)).sum().item()
                    batch_pred.append(pred_val.cpu().data.numpy())
                    batch_y_val_list.append(batch_y_val.cpu().data.numpy())
                    batch_pred_cate.append(output_val.cpu().data.numpy())

                val_loss /= len(X_val_input)

                val_loss_list.append(val_loss.cpu().data.numpy())
                # if batch_idx % 10 == 0:
                print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    val_loss, correct_val, len(X_val_input),
                    100. * correct_val / len(X_val_input)))

                yy_val = np.hstack(batch_y_val_list).reshape(-1, 1)
                ppred_classes = np.vstack(batch_pred)

                acc_val = accuracy_score(yy_val, ppred_classes)
                f1 = f1_score(yy_val, ppred_classes, average='micro')

                confusion_mat = metrics.confusion_matrix(yy_val, ppred_classes)
                mcc = metrics.matthews_corrcoef(yy_val, ppred_classes)

                encoder_ = LabelBinarizer()
                yy_val_ = encoder_.fit_transform(yy_val)
                roc_auc = metrics.roc_auc_score(yy_val_, np.exp(np.vstack(batch_pred_cate)), multi_class='ovr',
                                                average='micro')

        confusion_matrix_res.append(confusion_mat)
        mcc_res.append(mcc)
        acc_res.append(acc_val)
        auc_res.append(roc_auc)
        f1_res.append(f1)

        # torch.save(model, './pytorch_transformer_head' + str(n_head) + '_10label_zscore_1l_product_epoch'+str(epoch)+'_.model')
        torch.save(model, './model/1l/pytorch_transformer_head_' + str(n_head) + '_lr_' + str(lr_rate) + '_gain_' + str(
            gain) + '_34label_zscore_1l_product_' + str(act_fun) + '_epoch' + str(epoch) + '.model')
    res['confusion_matrix'] = confusion_matrix_res
    res['mcc'] = mcc_res
    res['f1'] = f1_res
    # res['sn'] = sn_res
    # res['sp'] = sp_res
    res['acc'] = acc_res
    res['auc'] = auc_res

    # torch.save(model,'./model/1l/pytorch_transformer_head_'+str(n_head)+'_lr_'+str(lr_rate)+'_gain_'+str(gain)+'_34label_zscore_1l_product_'+str(act_fun)+'.model')
    pl.dump(res, open('./model_res/1l/pytorch_transformer_head_' + str(n_head) + '_lr_' + str(lr_rate) + '_gain_' + str(
        gain) + '_34label_zscore_1l_product_' + str(act_fun) + '.dat', 'wb'))
    plt.plot(train_loss_list, label='train loss')
    plt.plot(val_loss_list, label='val loss')
    plt.legend()
    # plt.show()
    plt.savefig(
        'model_res/1l/pytorch_transformer_head_' + str(n_head) + '_lr_' + str(lr_rate) + '_gain_' + str(
            gain) + '_34label_zscore_1l_product_' + str(act_fun) + '.png', format='png')
    plt.close()
