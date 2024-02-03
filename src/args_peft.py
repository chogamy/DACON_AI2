import os 
import re
import yaml

def parse_peft(args, path):
    
    file_path = os.path.join(path, f"{args.peft}.yaml")
    
    if os.path.isfile(file_path):
        with open(file_path) as f:
            args.peft = yaml.full_load(f)
    else:
        print(file_path)
        raise ValueError('Pondering')
        
        # e = re.compile('e\d+')
        # b = re.compile('b\d+')
        # lr = re.compile('lr\d+e-\d+')

        # with open(os.path.join(path, 'basic.yaml')) as f:
        #     basic = yaml.full_load(f)
        
        # basic['num_train_epochs'] = int(e.findall(args.train)[0][1:])
        # basic['per_device_train_batch_size'] = int(b.findall(args.train)[0][1:])
        # basic['learning_rate'] = lr.findall(args.train)[0][2:]

        # with open(file_path, 'w') as f:
        #     yaml.dump(basic, f)      