echo " Running CIFAR10 Training EXP"

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 5 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 5 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 5 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 5 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 5 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only


python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 6 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 6 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 6 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 6 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only

python3 train.py --dataset 'CIFAR10' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 6 --mini 0.4 --pretrain 20 --gen_num 200 --std 2 --code_len 128 --alg 'FedProx' --prox_mu 0.001 --eval_only
