#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main.py \
#                     --mode train \
#                     --seed 42 \
#                     --post_train_path /root/DACON_AI2/checkpoints/post_train/maywell/Synatra-42dot-1.3B/ia3 \
#                     --data multi_train \
#                     --model maywell/Synatra-42dot-1.3B \
#                     --peft lora \
#                     --train e3b4lr5e-5 \

                    
CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode train \
                    --seed 42 \
                    --data multi_train \
                    --model TeamUNIVA/Komodo_7B_v1.0.0 \
                    --bnb true \
                    --peft lora \
                    --train e3b12lr5e-5 \
