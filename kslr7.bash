#!/bin/bash

# # 运行第一个学习率
python main_kslr.py --lr 0.001 --use_pretrain 2 --pretrain_model_path "kslr7/nce_para/best0.005.pth" --save_dir "kslr7/nce_para" \
                    --seed 42 --evaluate_every 10 --stopping_steps 2 --alpha 0.005 --beta 0.003 --cuda 1
# 运行第二个学习率
python main_kslr.py --lr 0.0001 --use_pretrain 1 --pretrain_model_path "kslr7/nce_para/best0.005.pth" --save_dir "kslr7/nce_para" \
                    --seed 42 --evaluate_every 1 --stopping_steps 10 --alpha 0.005 --beta 0.003 --cuda 1

