from peft import TaskType, get_peft_model

def peft(args, model):
    
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