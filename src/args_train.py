import os 
import re
import yaml

def parse_train(args, path):
    
    file_path = os.path.join(path, args.train)
    

    if os.path.isfile(file_path):
        with open(file_path) as f:
            args.train = yaml.full_load(f)
    else:
        args.train = args.train.split(".")[0]
        pattern = re.compile('e\d+b\d+lr\d+-\d+.yaml')
        

        with open(os.path.join(path, 'basic.yaml')) as f:
            args.train = yaml.full_load(f)
        
        
        # e3b2lr2e-5.yaml
    

    assert 1==0


    pass