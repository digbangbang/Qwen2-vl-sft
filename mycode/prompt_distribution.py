import json
from tqdm import tqdm

DATA_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data/viz_cls_2023_sharegpt.json'

'''
    >>> get the distribution of the prompt in viz
'''

def main():
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The JSON file should contain a list of items.")

    prompts = {}
    for i in tqdm(range(len(data))):
        category = data[i]['conversations'][0]['value']
        if category not in prompts.keys():
            prompts[category] = 1
        else:
            prompts[category] += 1

    with open('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data/data-distribution.json', 'w') as file:
        json.dump(prompts, file, ensure_ascii = False, indent = 4)

if __name__ == '__main__':
    main()