#!/bin/bash

# TeamUNIVA/Komodo_7B_v1.0.0
# maywell/Synatra-Mixtral-8x7B

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode post_train \
                    --seed 42 \
                    --data all_text \
                    --bnb true \
                    --model TeamUNIVA/Komodo_7B_v1.0.0 \
                    --peft lora \
                    --train e10b2lr5e-5 \

                    
