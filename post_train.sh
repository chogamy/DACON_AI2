#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode post_train \
                    --seed 42 \
                    --data all_text \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e3b2lr5e-5 \

                    