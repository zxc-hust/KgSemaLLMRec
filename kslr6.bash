#!/bin/bash

# 运行第一个学习率
python main_kslr.py --lr 0.001 --use_pretrain 0 --pretrain_model_path "kslr6/best.pth" --save_dir "kslr6/42" --seed 42
# 运行第二个学习率
python main_kslr.py --lr 0.0001 --use_pretrain 1 --pretrain_model_path "kslr6/best.pth" --save_dir "kslr6/42" --seed 42