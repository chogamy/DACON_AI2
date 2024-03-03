from peft import TaskType, get_peft_model
from peft import PeftConfig, PeftModel

def peft(args, model):
    if args.post_train_path is not None:
        model = PeftModel.from_pretrained(model, args.post_train_path)
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
    elif args.peft['name'] == 'ia3':
        from peft import IA3Config
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM, target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"])

        model = get_peft_model(model, peft_config)
        print(model.print_trainable_parameters())

    else:
        
        raise ValueError("Invalid PEFT")
    
    

    return model