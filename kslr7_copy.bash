#!/bin/bash

# 运行第一个学习率
python main_kslr.py --lr 0.001 --use_pretrain 2 --pretrain_model_path "kslr7/42_pre/best0.004.pth" --save_dir "kslr7/42_pre" \
                    --seed 42 --evaluate_every 10 --stopping_steps 2 --alpha 3 --beta 0.004
# 运行第二个学习率
python main_kslr.py --lr 0.0001 --use_pretrain 1 --pretrain_model_path "kslr7/42_pre/best0.004.pth" --save_dir "kslr7/42_pre" \
                    --seed 42 --evaluate_every 1 --stopping_steps 10 --alpha 3 --beta 0.004

