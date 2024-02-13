import os
from tqdm import tqdm

import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel

BATCH = True

def infer(args, model, tokenizer, dataset):
    test_dataset = load_dataset('text', data_files={'test': dataset})

    print(test_dataset)
    from pprint import pprint
    pprint(test_dataset['test']['text'][:5])
    print('-------------------------')

    model = PeftModel.from_pretrained(model, args.train['output_dir'])
    model.to('cuda')
    model.eval()

    tokenizer.padding_side = "left"


    if BATCH:
        batch_size = 8
        with open(os.path.join(args.train['output_dir'], 'output.txt'), 'w') as f:
            with torch.no_grad():
                for i in tqdm(range(0, len(test_dataset['test']['text']), batch_size), desc='infer'):
                    batch = test_dataset['test']['text'][i:i+batch_size]
                    inputs = tokenizer(batch, padding=True, return_tensors='pt')
                    outputs = model.generate(input_ids=inputs['input_ids'].to('cuda'), max_new_tokens=800)

                    outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)

                    for output in outputs:                    
                        q, a = output.split("답변:", 1)
                        # 안한게 더 좋네
                        # if "관계는 무엇인가요?" in q:
                        #     a = "모르겠습니다."
                        # else:
                        a = a.replace("\n", "")
                        a = a.strip()
                        f.write(f"{a}\n")
    else:
        with open(os.path.join(args.train['output_dir'], 'output.txt'), 'w') as f:
            with torch.no_grad():
                for line in tqdm(test_dataset['test']['text'], desc='infer'):
                    
                    line = line.strip()
                    line = line.replace("답변:", "")

                    qs = line.split('? ')
                    qs = [q.strip() for q in qs]
                    qs = list(filter(lambda x: x != '', qs))

                    for i in range(len(qs)):
                        if qs[i][-1] in ['.', '?']:
                            pass
                        else:
                            qs[i] = qs[i] + "?"
                    

                    if len(qs) < 1:
                        qs = qs[0].split('. ')

                        for i in range(len(qs)):
                            if qs[i][-1] in ['.', '?']:
                                pass
                            else:
                                qs[i] = qs[i] + "."

                    print(qs)
                    answers = []
                    
                    for q in qs:
                        if "관계는 무엇인가요?" in q:
                            a = "모르겠습니다."
                        else:
                            q = q + " 답변: "

                            inputs = tokenizer(q, return_tensors='pt')
                            outputs = model.generate(input_ids=inputs['input_ids'].to('cuda'), max_new_tokens=400)    
                            
                            outputs = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                            
                            q, a = outputs[0].split("답변: ", 1)
                            a = a.replace("답변:", "")
                            a = a.replace("질문:", "")
                            a = a.replace("\n", "")
                            a = a.strip()
                            
                        
                        answers.append(a)
                    
                    a = " 그리고 ".join(answers)
                    
                    f.write(f"{a}\n")
                    