
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel

def infer(args, model, tokenizer, dataset):
    test_dataset = load_dataset('text', data_files={'test': dataset})

    print(test_dataset)
    from pprint import pprint
    pprint(test_dataset['test']['text'][:5])

    model = PeftModel.from_pretrained(model, args.peft['output_dir'])
    
    with open("./output.txt", 'w') as f:
        with torch.no_grad():
            for line in test_dataset['test']['text']:
                inputs = tokenizer(line.strip(), return_tensors='pt')
                outputs = model.generate(input_ids=inputs['input_ids'], max_new_tokens=10)    
                
                outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                
                q, a = outputs[0].split("답변: ", 1)
                
                f.write(f"{a}\n")
    