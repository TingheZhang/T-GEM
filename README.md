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

## 2 train the model 
G-TEM_pytorch_3l_34.py is the model that has the best performance for our cancer prediction task. The code can sinply run by :
> python G-TEM_pytorch_3l_34.py 5 0.001 relu 
which 5 is the number of head for each attention layer, 0.001 is the learning rate, relu is the activation function used for last attention layer. (you can choose the actvation form relu, leakyrelu,gelu or nan where nan means not using any activation function.)


Model structure can be changed on line 193~ line 208. 
## 3 compute the attribution score 
To evaluate which gene is more important to predicte specific cancer, we use integer gradient(IG) to compute the attribution score for each test samples. The larger score means more importance. 
To compute the attribution score, it can be easily got by:
> python G-TEM_t_attr_allcancer.py ig 
The matched parameter need to be adjust for different project at line 35 ~ line 51. 

## 4


