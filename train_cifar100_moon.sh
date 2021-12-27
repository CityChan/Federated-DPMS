echo " Running CIFAR100 Training EXP"



python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 1

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.1 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 1

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.2 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 5

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 1

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.3 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 5
n_mu 10

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 1

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.4 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 5


python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 1

python3 train.py --dataset 'CIFAR100' --batch_size 64 --lr 0.001 --num_epochs 50 --dropout_rate 0.2 --tag 'federated' --num_users 10 --update_frac 1 --local_ep 5 --beta 0.5 --seed 2 --mini 1 --pretrain 20 --gen_num 50 --std 4 --code_len 128 --alg 'Moon' --moon_mu 5

