echo " Running CIFAR100 Training EXP"

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedAvg'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedVAE'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 10 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'Moon' --moon_mu 10

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedAvg'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedVAE'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 10 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'


python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'Moon' --moon_mu 10

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedAvg'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedVAE'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 10 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'Moon' --moon_mu 10

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedProx' --prox_mu 0.001


python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedAvg'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedVAE'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 10 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'Moon' --moon_mu 10

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedProx' --prox_mu 0.001


python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedAvg'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedVAE'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 10 --gen_num 200 --std 4 --code_len 32 --alg 'DPMS'

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'Moon' --moon_mu 10

python3 train.py --dataset 'FMNIST' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 200 --std 4 --code_len 32 --alg 'FedProx' --prox_mu 0.001



