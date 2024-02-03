import os
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


DIR = os.path.dirname(os.path.realpath(__file__))

def peft(args, model):
    from peft import TaskType, get_peft_model

    if args.peft == 'lora':
        from peft import LoraConfig
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
    elif args.peft == None:
        pass
    else:
        raise ValueError("Invalid PEFT")
    
    if args.peft == None:
        pass
    else:
        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str, required=True, choices=['train', 'infer'])
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--peft", default=None, required=True, choices=['lora'])
    parser.add_argument("--train", default=None, required=True)

    args = parser.parse_args()

    
    # set seed

    if args.mode == 'train':
        from src.train import train

        train()

        assert 1==0
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        model = peft(args, model)

        train_dataset = load_dataset('text', data_files={'train': os.path.join(DIR, 'data', 'train.txt')})

        print(train_dataset)
        from pprint import pprint
        pprint(train_dataset['train']['text'][:5])

        def tokenize_function(examples):
            return tokenizer(examples['text'].strip())
    
        tokenized_datasets = train_dataset.map(tokenize_function, 
                                            batched=True, 
                                            remove_columns=["text"], 
                                            load_from_cache_file=False, 
                                            desc="Data Pre-processing")


        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=1,  # batch size per device during training
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
        )


        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_datasets['train'],         # training dataset
            data_collator=data_collator,         # collator to use for training
        )

        trainer.train()
        args.peft['save_path']
        # model.save()
    elif args.mode == 'infer':
        pass
    else:
        raise ValueError('Invalid mode')
        