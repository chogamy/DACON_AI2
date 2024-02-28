#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode infer \
                    --seed 42 \
                    --data test \
                    --path /root/DACON_AI2/checkpoints/train/maywell/Synatra-42dot-1.3B/lora \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e3b2lr2e-5 \