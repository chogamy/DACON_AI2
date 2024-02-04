#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
                    --mode infer \
                    --seed 42 \
                    --model maywell/Synatra-42dot-1.3B \
                    --peft lora \
                    --train e2b16lr2e-5 \