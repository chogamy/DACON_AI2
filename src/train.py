import json

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from .peft import peft



def train(args, model, tokenizer, dataset):
    train_dataset = load_dataset('text', data_files={'train': dataset})

    print(train_dataset)
    from pprint import pprint
    pprint(train_dataset['train']['text'][:5])

    def tokenize_function(examples):
            return tokenizer(examples['text'])
    
    tokenized_datasets = train_dataset.map(tokenize_function, 
                                        batched=True, 
                                        remove_columns=["text"], 
                                        load_from_cache_file=False, 
                                        desc="Data Pre-processing")
    

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    model = peft(args, model)
    
    if args.ds_config is not None:
        with open(args.ds_config, 'r') as f:
            args.ds_config = json.load(f)
    
    training_args = TrainingArguments(
            output_dir=args.train['output_dir'],
            logging_dir=args.train['logging_dir'],
            logging_strategy=args.train['logging_strategy'],
            learning_rate=args.train['learning_rate'],
            lr_scheduler_type=args.train['lr_scheduler_type'],
            optim=args.train['optim'],
            save_strategy=args.train['save_strategy'],
            do_train=args.train['do_train'],
            gradient_checkpointing=args.train['gradient_checkpointing'],
            num_train_epochs=args.train['num_train_epochs'],
            per_device_train_batch_size=args.train['per_device_train_batch_size'],
            warmup_steps=args.train['warmup_steps'],
            weight_decay=args.train['weight_decay'],
            deepspeed=args.ds_config
        )

    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.peft['output_dir']) 
    
