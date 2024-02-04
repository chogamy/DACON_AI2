import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.args_train import parse_train
from src.args_peft import parse_peft

DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str, required=True, choices=['train', 'infer'])
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--peft", default='lora', required=True, choices=['lora'])
    parser.add_argument("--train", default=None, required=True)

    args = parser.parse_args()
    
    parse_train(args, os.path.join(DIR, 'args', 'train'))
    parse_peft(args, os.path.join(DIR, 'args', 'peft'))

    # set seed
    args.train['seed'] = args.seed

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.mode == 'train':
        from src.train import train

        dataset = os.path.join(DIR, 'data', 'train.txt')

        train(args, model, tokenizer, dataset)

        
    elif args.mode == 'infer':
        from src.train import infer

        dataset = os.path.join(DIR, 'data', 'test.txt')

        infer(args, model, tokenizer, dataset)
        
    else:
        raise ValueError('Invalid mode')
        