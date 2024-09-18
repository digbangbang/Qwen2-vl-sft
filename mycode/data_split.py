import json

DATA_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data/viz_cls_2023_sharegpt.json'

'''
    >>> split the dataset into small pieces
    >>> cut the dataset uniformly into $num_files
'''

def main(num_files = 10):
    with open(DATA_PATH, 'r') as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("The JSON file should contain a list of items.")

    num_items_per_file = len(data) // num_files
    remainder = len(data) % num_files
    for i in range(num_files):
        start_index = i * num_items_per_file + min(i, remainder)
        end_index = start_index + num_items_per_file + (1 if i < remainder else 0)
        
        sliced_data = data[start_index:end_index]
        
        with open(f'/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/data/viz_{i+1}.json', 'w') as outfile:
            json.dump(sliced_data, outfile, indent=4)

if __name__ == '__main__':
    main(num_files = 10)