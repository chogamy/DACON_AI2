import os
from tqdm import tqdm

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel

def infer(args, model, tokenizer, dataset):
    test_dataset = load_dataset('text', data_files={'test': dataset})

    print(test_dataset)
    from pprint import pprint
    pprint(test_dataset['test']['text'][:5])

    model = PeftModel.from_pretrained(model, args.train['output_dir'])
    model.to('cuda')
    model.eval()

    with open(os.path.join(args.train['output_dir'], 'output.txt'), 'w') as f:
        with torch.no_grad():
            for line in tqdm(test_dataset['test']['text'], desc='infer'):
                inputs = tokenizer(line.strip(), return_tensors='pt')
                outputs = model.generate(input_ids=inputs['input_ids'].to('cuda'), max_new_tokens=10)    
                
                outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                
                q, a = outputs[0].split("답변: ", 1)
                
                f.write(f"{a}\n")
    