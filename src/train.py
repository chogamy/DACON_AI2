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

    training_args = TrainingArguments(
            output_dir=args.train['output_dir'],
            logging_dir=args.train['logging_dir'],
            logging_strategy=args.train['logging_strategy'],
            save_strategy=args.train['save_strategy'],
            do_train=args.train['do_train'],
            num_train_epochs=args.train['num_train_epochs'],
            per_device_train_batch_size=args.train['per_device_train_batch_size'],
            warmup_steps=args.train['warmup_steps'],
            weight_decay=args.train['weight_decay'],
        )

    

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_datasets['train'],         # training dataset
        data_collator=data_collator,         # collator to use for training
    )

    trainer.train()
    model.save_pretrained(args.peft['output_dir']) 
    
