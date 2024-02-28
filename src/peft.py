from peft import TaskType, get_peft_model
from peft import PeftConfig, PeftModel

def peft(args, model):
    if args.path is not None:
        model = PeftModel.from_pretrained(model, args.path)
        model = model.merge_and_unload()

    if args.peft['name'] == None:
        return model
    elif args.peft['name'] == 'lora':
        from peft import LoraConfig
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.peft['r'], lora_alpha=args.peft['alpha'], lora_dropout=args.peft['dropout']
        )

        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())
    else:
        
        raise ValueError("Invalid PEFT")
    
    

    return model