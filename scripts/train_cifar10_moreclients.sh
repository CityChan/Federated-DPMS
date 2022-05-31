echo " Running CIFAR10 Training EXP"


python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.1 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedAvg'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.1 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedVAE'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.1 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'DPMS'


python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.1 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.1 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 20 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.2 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedAvg'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 20 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.2 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedVAE'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 20 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.2 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'DPMS'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 20 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.2 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 20 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.2 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 30 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.3 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedAvg'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 30 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.3 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedVAE'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 30 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.3 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'DPMS'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 30 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.3 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 30 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.3 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 40 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.4 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedAvg'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 40 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.4 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedVAE'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 40 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.4 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'DPMS'


python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 40 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.4 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 40 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.4 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedProx' --prox_mu 0.001

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 50 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.5 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedAvg'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 50 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.5 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedVAE'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 50 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.5 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'DPMS'

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 50 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.5 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 50 --update_frac 1 --local_ep 5 --beta 0.5 --seed 1 --mini 0.5 --pretrain 25 --gen_num 100 --std 3 --code_len 128 --alg 'FedProx' --prox_mu 0.001


