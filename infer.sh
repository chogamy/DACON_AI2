#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode infer \
                    --seed 42 \
                    --data test \
                    --post_train_path /root/DACON_AI2/checkpoints/post_train/maywell/Synatra-42dot-1.3B/ia3 \
                    --train_path /root/DACON_AI2/checkpoints/train/maywell/Synatra-42dot-1.3B/ia3 \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e3b4lr5e-5 \