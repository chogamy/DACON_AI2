import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.args_train import parse_train
from src.args_peft import parse_peft

DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str, required=True, choices=['post_train', 'train', 'infer'])
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--post_train_path", default=None, type=str, required=False)
    parser.add_argument("--train_path", default=None, type=str, required=False)
    parser.add_argument("--data", default=None, type=str, required=True, choices=['multi_train', 'all_text', 'train', 'test'])
    parser.add_argument("--peft", default='lora', required=True, choices=['lora', 'none', 'ia3']) 
    parser.add_argument("--bnb", default=True, required=True) 
    parser.add_argument("--train", default=None, required=True)


    #########################
    parser.add_argument("--local_rank", default=None, required=False)
    parser.add_argument("--ds_config", default=None, required=False)

    args = parser.parse_args()
    
    parse_train(args, os.path.join(DIR, 'args', 'train'))
    parse_peft(args, os.path.join(DIR, 'args', 'peft'))

    # set seed
    args.train['seed'] = args.seed

    bnb_config=None
    if args.bnb:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
            )
        

    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))


    if args.mode in ['train', 'post_train']:
        from src.train import train

        dataset = os.path.join(DIR, 'data', f"{args.data}.txt")

        train(args, model, tokenizer, dataset)

        
    elif args.mode == 'infer':
        from src.infer import infer

        dataset = os.path.join(DIR, 'data', 'test.txt')

        infer(args, model, tokenizer, dataset)
        
    else:
        raise ValueError('Invalid mode')
        