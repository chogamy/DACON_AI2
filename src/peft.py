from peft import TaskType, get_peft_model

def peft(args, model):
    
    if args.peft['name'] == 'lora':
        from peft import LoraConfig
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft['r'], lora_alpha=args.peft['alpha'], lora_dropout=args.peft['dropout']
        )
    else:
        
        raise ValueError("Invalid PEFT")
    
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    return model