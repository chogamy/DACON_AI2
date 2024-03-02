#!/bin/bash

deepspeed main.py \
        --mode post_train \
        --seed 42 \
        --data all_text \
        --model maywell/Synatra-42dot-1.3B \
        --peft lora \
        --train e10b2lr5e-5 \
        --ds_config /root/DACON_AI2/args/ds_confg.json \

                    
