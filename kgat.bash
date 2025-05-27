#!/bin/bash

# 运行第一个学习率
python main_kgat.py --lr 0.001 --use_pretrain 0 --save_dir "kgat/42" --pretrain_model_path "trained_model/KGAT/amazon-book/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.001_pretrain0/model_epoch17.pth" --seed 42

# 运行第二个学习率
python main_kgat.py --lr 0.0001 --use_pretrain 1 --pretrain_model_path "kgat/42/best.pth" --save_dir "kgat/42" --seed 42