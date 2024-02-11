import os
import pandas as pd


DIR = os.getcwd()

if __name__ == "__main__":
    print(DIR)

    test_dataset = pd.read_csv(os.path.join(DIR, 'DACON_AI2', 'data', 'test.csv'))

    for item in test_dataset['질문']:
        item = item.replace('"', '') 
        print("원래 질문--------------------------")
        print(item)

        
        lines = item.split('? ')
        lines = [line.strip() for line in lines]
        lines = list(filter(lambda x: x != '', lines))
        if len(lines) > 1:
            print("분리 질문--------------------------")
            for line in lines:
                if line[-1] == '?':
                    print(line)
                else:
                    print(f"{line}?")
            
            breakpoint()
        else:
            lines = lines[0].split('. ')
            print("분리 질문--------------------------")
            for line in lines:
                if line[-1] == '?':
                    print(line)
                elif line[-1] == '.':
                    print(line)
                else:
                    print(f"{line}.")
            
            breakpoint()
