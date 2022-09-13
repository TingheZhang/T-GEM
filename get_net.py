import pickle as pl
import seaborn,argparse,os
import matplotlib.pyplot as plt
import numpy as np



parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
	"--head_num", default=None, type=int, required=True, help="The number of head for each layers"
)
parser.add_argument(
	"--cancer_type", default=None, type=int, required=True, help="Compute the attribution for which cancer, please check the table for exact cancer number"
)
parser.add_argument(
	"--threshold", default=None, type=float, required=True, help="select genes links with top attribution score"
)

# Other parameters

parser.add_argument(
	"--result_dir",
	required=True,
	type=str,
	help="The dir used to  save result",
)
parser.add_argument(
	"--net_dir",
	required=True,
	type=str,
	help="The dir used to  save gene link",
)

args = parser.parse_args()
if not os.path.exists(args.net_dir):
    os.makedirs(args.net_dir, exist_ok=True)

n_gene = 1708
n_feature = 1708
n_head=args.head_num
cancer_type=args.cancer_type
n_layer=3
y, data_df, pathway_gene, pathway, cancer_name = pl.load(open('./pathway_data.pckl', 'rb'))
data_ = np.array(data_df)
x = np.float32(data_)
genes = np.array(data_df.columns.tolist())

test_softmax_attr_l1=pl.load(open(args.result_dir+'/attr_softmax_alllayer_'+str(cancer_type)+'.plk','rb'))[0]
test_softmax_attr_l1_=test_softmax_attr_l1.reshape(n_head,n_gene*n_feature)

test_softmax_attr_l2=pl.load(open(args.result_dir+'/attr_softmax_alllayer_'+str(cancer_type)+'.plk','rb'))[1]
test_softmax_attr_l2_=test_softmax_attr_l2.reshape(n_head,n_gene*n_feature)

test_softmax_attr_l3=pl.load(open(args.result_dir+'/attr_softmax_alllayer_'+str(cancer_type)+'.plk','rb'))[2]
test_softmax_attr_l3_=test_softmax_attr_l3.reshape(n_head,n_gene*n_feature)

attr_one_mean=np.vstack((test_softmax_attr_l1,test_softmax_attr_l2,test_softmax_attr_l3))
attr_one_mean=np.sum(attr_one_mean,axis=1)
link_threshold=[]


attr_one_mean_norm=[]
for layer in range(n_layer):
    # attr_one_mean_norm.append(attr_one_mean[layer]/np.max(attr_one_mean[layer]))
    attr_one_mean_norm.append(attr_one_mean[layer] / np.max(abs(attr_one_mean[layer])))
attr_one_mean_norm_array=np.array(attr_one_mean_norm)
attr_one_mean_norm_array_sorted_index=np.argsort(-attr_one_mean_norm_array.flatten())
threshold=attr_one_mean_norm_array.flatten()[attr_one_mean_norm_array_sorted_index[np.int(len(attr_one_mean_norm_array_sorted_index)*args.threshold)]]
link_selected_threshold=np.nonzero(attr_one_mean_norm_array > threshold)
link_threshold.append(link_selected_threshold)

for layer in range(n_layer):
        ind_layer = np.where(link_threshold[0][0] == layer)
        query_link=link_threshold[0][1][ind_layer]
        key_link = link_threshold[0][2][ind_layer]

        net_save=np.vstack((genes[query_link],genes[key_link])).T
        np.savetxt(args.net_dir+'/link_layer_' + str(layer) +'_net.txt',
                   net_save, fmt='%s',delimiter='\t')

        # att_weights = attr_one_mean[layer][query_link, key_link]
        # att_weights_norm = att_weights/np.max(attr_one_mean)
        # net_save = np.vstack((genes[query_link], genes[key_link], att_weights_norm)).T
        # np.savetxt(args.net_dir+'/link_layer_' + str(layer)  +'_net_weighted.txt',
        #            net_save, fmt='%s',delimiter='\t')
