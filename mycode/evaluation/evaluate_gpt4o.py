import os
import openai
import traceback
import time
import json
import base64
import random
import logging

from tqdm import tqdm

DATA_ROOT = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/ruandelian/autovision_smart_anno_data/mllm_data'

random.seed(0)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/LLaMA-Factory/mycode/logger/gpt4o.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def get_gpt4_result(content):
    openai.api_key = "1648285843187896359"
    openai.api_base = "https://aigc.sankuai.com/v1/openai/native"

    try_cnt = 3
    while try_cnt:
        try:
            result = openai.ChatCompletion.create(
                model="gpt-4o-2024-05-13",
                user="1",
                messages=[
                        {"role": "user", "content": content},
                ]
            )
            break
        except:
            result = ''
            try_cnt -= 1
            traceback.print_exc()
            print(f'try_cnt=={try_cnt}')
            time.sleep(2)

    return result.to_dict()['choices'][0]['message']['content'] if result != '' else ''


def parse_data(img_pa):
    anno_file = os.path.join(img_pa, "train_annotation.json")
    data = []

    with open(anno_file, 'r') as f:
        lines = json.load(f)
        try:
            num_samples = min(len(lines), 100)
            selected_lines = random.sample(lines, num_samples)
            for line in selected_lines:
                if '.gif' in line['imageurl']:
                    filename = str(line['imageId']) + '.gif'
                else:
                    filename = str(line['imageId']) + '.jpg'
                if not os.path.exists(f"{img_pa}/{filename}"):
                    continue
                file_attr = list(json.loads(line['annotation'])['file_attributes'].values())[0]
                if isinstance(file_attr, dict):
                    answer = list(file_attr.keys())[0]
                elif isinstance(file_attr, str):
                    answer = file_attr
                if answer == "":
                    continue
                
                new_line = {
                    'index': line['imageId'],
                    'image': f"{img_pa}/{filename}",
                    'answer': answer
                }
                data.append(new_line)
        except Exception as es:
                print('!!!标签文件{}解析异常: {}'.format(anno_file, es))
    return data


def get_prompt(ds_path):
    anno_tree = json.load(open(os.path.join(ds_path, "anno.txt"), "r"))
    print("anno_tree: ", anno_tree)
    logger.info(f'anno tree {anno_tree}')
    anno_key = list(anno_tree.keys())[0]
    task_comment = ""
    if "task_comment" in anno_tree:
        task_comment = anno_tree["task_comment"]
    anno_list = ""
    comment_list = ""
    for anno_name, anno_comment in anno_tree[anno_key].items():
        anno_list += f"<{anno_name}>，"
        if anno_comment != "":
            comment_list += f"<{anno_name}>表示{anno_comment}。\n"
    anno_list = anno_list[:-1]
    if comment_list == "" and task_comment == "":
        prompt = f"假如你是一个AI标注员，请根据我的提示信息，直接输出已定义好格式的答案。\n现在进行{anno_key}相关的图片标注。请仔细观察图片，并给图片打上如下标签之一。\n<|image|>\n<标签集合>：{anno_list}。\n输出格式：x，其中x为<标签集合>中的一项。"
    elif comment_list == "":
        prompt = f"假如你是一个AI标注员，请根据我的提示信息，直接输出已定义好格式的答案。\n现在进行{anno_key}相关的图片标注，{task_comment}。请仔细观察图片，并给图片打上如下标签之一。\n<|image|>\n<标签集合>：{anno_list}。\n输出格式：x，其中x为<标签集合>中的一项。"
    elif task_comment == "":
        prompt = f"假如你是一个AI标注员，请根据我的提示信息，直接输出已定义好格式的答案。\n现在进行{anno_key}相关的图片标注。请仔细观察图片，并给图片打上如下标签之一。\n<|image|>\n<标签集合>：{anno_list}。\n标签说明：\n{comment_list}\n输出格式：x，其中x为<标签集合>中的一项。"
    else:
        prompt = f"假如你是一个AI标注员，请根据我的提示信息，直接输出已定义好格式的答案。\n现在进行{anno_key}相关的图片标注，{task_comment}。请仔细观察图片，并给图片打上如下标签之一。\n<|image|>\n<标签集合>：{anno_list}。\n标签说明：\n{comment_list}\n输出格式：x，其中x为<标签集合>中的一项。"
        
    prompt = f"USER: {prompt} ASSISTANT: "
    print("prompt: ", prompt)
    return prompt


def encode_image(img_path):
    with open(img_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def check(ds_names):
    exist_names = []
    for name in ds_names:
        if os.path.exists(os.path.join(DATA_ROOT, name)):
            exist_names.append(name)
    return exist_names


def main():
    data_list = os.path.join(DATA_ROOT, "data_list_test_comment.txt")
    data_lines = open(data_list, 'r').readlines()
    ds_names = []
    for idx, line in enumerate(data_lines):
        try:
            task_id, ds_name, ds_version, anno = line.strip().split("\t")
            ds_names.append(f"{ds_name}_{ds_version}")
        except:
            print("error: files went wrong!")

    ds_names = check(ds_names)

    logger.info(f'total datasets {len(ds_names)}')
    for ds_name in ds_names:
        total = 0
        right_total = 0
        di_ri = {}
        di_to = {}
        print(ds_name)
        logger.info(f'Now test {ds_name}')
        dataset_path = os.path.join(DATA_ROOT, ds_name)
        data = parse_data(dataset_path)
        base64_images = [encode_image(img['image']) for img in data]
        answers = [da['answer'] for da in data]
        prompt = get_prompt(dataset_path)

        for i in tqdm(range(len(base64_images))):
            base64_image = [
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{base64_images[i]}',
                        'detail': 'high',
                    },
                }
            ]
            user_content = [{'type': 'text', 'text': prompt}]
            
            user_content.extend(base64_image)

            pred = get_gpt4_result(user_content)
            pred = pred.replace('<','').replace('>','')

            total += 1
            right_total += int(pred == answers[i])

            if answers[i] == pred:
                if answers[i] not in di_ri.keys():
                    di_ri[answers[i]] = 1
                else:
                    di_ri[answers[i]] += 1
            if answers[i] not in di_to.keys():
                di_to[answers[i]] = 1
            else: 
                di_to[answers[i]] += 1
        
        print("total {}, right {}, acc {}".format(total, right_total, round(right_total * 100.0 / total, 2)))
        logger.info("total {}, right {}, acc {}".format(total, right_total, round(right_total * 100.0 / total, 2)))
        for _ in di_to.keys():
            if _ not in di_ri.keys():
                print(f'{_} was not predicted, total {di_to[_]}')
                logger.info(f'{_} was not predicted, total {di_to[_]}')
                continue
            print(f'{_} acc {di_ri[_] / di_to[_]} total {di_to[_]}')   
            logger.info(f'{_} acc {di_ri[_] / di_to[_]} total {di_to[_]}')
        logger.info('\n')


if __name__ == '__main__':
    main()

