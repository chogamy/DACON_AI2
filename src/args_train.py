import os 
import re
import yaml

def parse_train(args, path):
    
    file_path = os.path.join(path, f"{args.train}.yaml")
    
    if os.path.isfile(file_path):
        with open(file_path) as f:
            args.train = yaml.full_load(f)
    else:
        
        e = re.compile('e\d+')
        b = re.compile('b\d+')
        lr = re.compile('lr\d+e-\d+')

        with open(os.path.join(path, 'basic.yaml')) as f:
            basic = yaml.full_load(f)
        
        basic['num_train_epochs'] = int(e.findall(args.train)[0][1:])
        basic['per_device_train_batch_size'] = int(b.findall(args.train)[0][1:])
        basic['learning_rate'] = lr.findall(args.train)[0][2:]

        with open(file_path, 'w') as f:
            yaml.dump(basic, f)

        args.train = basic