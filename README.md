# Federated Learning in Non-IID Settings Aided by Differentially Private Synthetic Data
This is an official repository for our CVPR2023 workshop paper
* "[Federated Learning in Non-IID Settings Aided by Differentially Private Synthetic Data](https://citychan.github.io/assets/publications/2023_cvpr/paper.pdf)"


<figure>
  <p align="center">
  <img src="img/DPMS.png" width=90% align="center" alt="my alt text"/>
  </p>
  <figcaption width=80%><em>
  FedDPMS and synthetic data generation. The four parts of the figure depict: (1) finding latent representation of raw data via a local encoder; (2) creating noisy latent means (by adding Gaussian noise to the means of latent data representations) and filtering out unusable ones with the help of a local classifier; (3) uploading usable noisy latent means to the server; (4) a benefiting client utilizing the global decoder to generate synthetic data from the received noisy latent means, expanding its local dataset.
  </em></figcaption>
</figure>
<br/>
<br/>



### Environment 
This project is developed based on python 3.6 with [torch1.9 (rocm4.2)](https://pytorch.org/get-started/previous-versions/). We use [conda](https://www.anaconda.com/docs/main) to manage the virtual environment.
```
git clone git@github.com:CityChan/Federated-DPMS.git
cd Federated-DPMS
conda create -n dpms --python=3.6
conda activate dpms
pip install torch==1.9.1+rocm4.2 torchvision==0.10.1+rocm4.2 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

### Code structure 
* `train.py`: general setup for training and evaluation
* `models.py`: model architectures for running experiments
* `sampling.py`: functions for generating non-iid datasets for federated learning
* `util.py`: functions for computing accuracy, knowledge distillation and model aggregation
* `Localupdate.py`: define functions for locally updating models with FedAvg, FedProx, Moon, FedMix and FedDPMS

#### Parameters
* --dataset: 'CIFAR10', 'CIFAR100', 'FMNIST'
* --batch_size: 64 by default 
* --num_epochs: number of global rounds, 50 by default
* --lr: learning rate, 0.001 by default
* --lr_sh_rate: period of learning rate decay, 10 by default
* --dropout_rate: drop out rate for each layer, 0.2 by default
* --tag: 'centralized', 'federated'
* --num_users: number of clients, 10 by default
* --update_frac: proportion of clients send updates per round, 1 by default
* --local_ep: local epoch, 5 by default
* --beta: concentration parameter for Dirichlet distribution: 0.5 by default
* --seed: random seed(for better reproducting experiments): 0 by default
* --mini： use part of samples in the dataset: 1 by default
* --moon_mu: hyper-parameter mu for moon algorithm, 5 by default
* --moon_temp: temperature for moon algorithm, 0.5 by default
* --prox_mu： hyper-parameter mu for prox algorithm, 0.001 by default
* --pretrain： number of preliminary rounds, 20 by default
* --gen_num: desired generation number for each class, 50 by default
* --std: standard deviation by Differential Noise, 4 by default
* --code_len: length of latent vector, 32 by default
* --alg: 'FedAvg, FedProx, Moon, FedVAE, DPMS, FedMix'
* --vae_mu: hyper-parameter for FedVAE and FedDPMS: 0.05 by default
* --fedmix_lam: lambda for fedmix: 0.05 by default
* --eval_only: only ouput the testing accuracy during training and the running time

#### Running the code for training and evaluation
We mainly use a .sh files to execute multiple expriements in parallel. 
The exprimenets are saved in checkpoint with unique id. Also, when the dataset is downloaded for the first time it takes a while. 

example: 

(1) for training a DPMS model   
```
python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 0 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'DPMS' --vae_mu 0.05
```

(2) for test the trained and saved model   
```
python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 0 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'DPMS' --vae_mu 0.05 --eval_only
```

You can explore the different .sh files in the 'scripts' folder for more examples.

####  Visualization of experiment results
<div align='center'>
<img src="img/accuracy_fmnist.png"></img>
</div>
<br />
<div align='center'>
<img src="img/accuracy_cifar10.png"></img>
</div>
<br />
<div align='center'>
<img src="img/accuracy_cifar100.png"></img>
</div>
<br />
<div align='center'>
<img src="img/fmnist_acc.png"></img>
</div>
<br />
<div align='center'>
<img src="img/cifar10_acc.png"></img>
</div>
<br />
<div align='center'>
<img src="img/fmnist_acc.png"></img>
</div>
<br />

#### Citation
We appreciate your citation if you use this codebase.
```
@article{chen2022federated,
  title={Federated Learning in Non-IID Settings Aided by Differentially Private Synthetic Data},
  author={Chen, Huancheng and Vikalo, Haris},
  journal={arXiv preprint arXiv:2206.00686},
  year={2022}
}
```
