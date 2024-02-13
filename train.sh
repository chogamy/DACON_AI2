#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode train \
                    --seed 42 \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e3b4lr2e-5 \
