# T-GEM
This repository includes the implementation of 'Transformer for Gene Expression Modeling (T-GEM): An interpretable deep learning model for gene expression-based phenotype predictions'. 

Please cite our paper if you use the models or codes. The repo is still actively under development, so please kindly report if there is any issue encountered.

## Citation
Ting-He Zhang, Md Musaddaqui Hasib, Yu-Chiao Chiu, et al. ransformer for Gene Expression Modeling (T-GEM): An interpretable deep learning model for gene expression-based phenotype predictions. 

## 1. Environment setup 
[Conda](https://docs.anaconda.com/anaconda/install/linux/) is recommanded to set up the enviroment. 
you can simply install the necessary dependacy by using command 

  > conda env create -f transformer.yml 

you can set your enviroment names by change the first line of the transformer.yml . Details can be found at [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

The following package is necessary for our project: pytorch=1.9.0, captum=0.4.0,python=3.9
After enviroment setup, use following command to activate this enviroments:
 > conda activate transformer

## 2. Train the model 
G-TEM_pytorch_3l_34.py is the model that has the best performance for our cancer prediction task. The code can sinply run by :
> python G-TEM_pytorch_3l_34.py  --head_num 5 --learning_rate 0.0001 --act_fun leakyrelu --batch_size 16 --epoch 1 --do_val --dropout_rate 0.3 --result_dir model_res/3l/ --model_dir model/3l/

The code attached used 3 attention layers. If you want to increase or decrease the number of layers, you can change the structure at line 252~ line 255 and line 271~ line 286. 

Our input file can be downloaded from [here](https://drive.google.com/file/d/13-Xjqexsi8-ZkZm17vcH6oGIivL2O8XW/view?usp=sharing)
## 3. Compute the attribution score 
To evaluate which gene is more important to predicte specific cancer, we use integer gradient(IG) to compute the attribution score for each test samples. The larger score means more importance. 
To compute the attribution score, it can be easily got by:
> python G-TEM_t_attr_allcancer.py ig 
> python G-TEM_t_attr_allcancer.py --head_num 5 --learning_rate 0.0001 --act_fun leakyrelu --batch_size 16 --epoch 1 --do_val --dropout_rate 0.3 --result_dir attr_34cancer/ --attr_method ig --model_location model/3l/pytorch_transformer_head_5_lr_0.0001_leakyrelu_epoch0.model --abs 

The matched parameter need to be adjust for different project at line 35 ~ line 51. Notice, the model structure in G-TEM_t_attr_allcancer.py has to match the structure at G-TEM_pytorch_3l_34.py .  The directory of trained model should be added at line 318


Gene symbol and its attribution score can be got for each cancer after running the program. 
## 4. Compute and visulize the attention weights and entropy for each attention layers
To discover the inner relationship at each layer, we can use attention weights and entropy of attention wieghts.  
G-TEM_t_vis.py has two mode 'attn' and 'vis'. 'attn' is used for get the attn value for each layer
> python G-TEM_t_vis.py attn


'vis' for visualizing the result from attn step
> python G-TEM_t_vis.py vis 

The model for G-TEM_t_vis.py has to match the one trained at step 2
