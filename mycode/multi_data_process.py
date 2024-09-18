import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
from transformers.image_utils import get_image_size, to_numpy_array
from transformers.image_transforms import convert_to_rgb

'''
    >>> transfer data format from mPLUG-OWL2 json to sharegpt json
    
    >>> <|image|> -> <image>
    >>> id -> null
    >>> "image": "XX" -> "image": ["XX"]
    
    >>> Select the .jpg that width and length > 28, width and length <= 2000
'''

DATA_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/dataset/viz/'
DATA_FILE = 'viz_cls_2023.json'
OUTPUT_FILE = 'viz_cls_2023_sharegpt.json'
IMAGE_PATH = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/ruandelian/autovision_smart_anno_data/mllm_data/'


def check_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            img = convert_to_rgb(img)
            img_np = to_numpy_array(img)
            height, width = get_image_size(img_np)
            if img_np.shape[-1] != 3:
                return False
            if height > 2000 or width > 2000:
                return False
            if width > 200 and height > 200:
                return True
            return False
            
    except IOError:
        print(f"Error opening or processing image file {image_path}.")
        return False
    

def process_item(item):
    if "id" in item:
        del item["id"]

    for conversation in item.get("conversations", {}):
        conversation["value"] = conversation["value"].replace("<|image|>", "<image>")

    item["images"] = [IMAGE_PATH + item["image"]]
    del item["image"]

    if all(check_image_size(img_path) for img_path in item["images"]):
        return item
    return None


def main():
    file_path = DATA_PATH + DATA_FILE
    output_path = DATA_PATH + OUTPUT_FILE

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    with Pool(processes=cpu_count()) as pool:
        results = pool.imap(process_item, data)
        valid_data = []
        for item in tqdm(results, total=len(data), desc='Processing items'):
            if item is not None:
                valid_data.append(item)
    print(len(valid_data))

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(valid_data, file, indent=4, ensure_ascii=False)

    print("JSON file has been processed and saved.")


if __name__ == "__main__":
    main()