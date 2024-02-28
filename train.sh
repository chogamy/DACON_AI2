#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode train \
                    --seed 42 \
                    --path /root/DACON_AI2/checkpoints/post_train/maywell/Synatra-42dot-1.3B/lora/checkpoint-2892 \
                    --data multi_train \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e3b4lr2e-5 \

                    
