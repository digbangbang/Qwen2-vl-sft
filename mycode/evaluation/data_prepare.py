import os
import json
import torch
import random

from log import get_logger
from functools import partial
from qwen_vl_utils import process_vision_info

logger = get_logger()

def collate_fn(batches, processor):
    questions = [_['questions'] for _ in batches]
    indices = [_['index'] for _ in batches]
    gts = [_['answers'] for _ in batches]
    images = [_['images'] for _ in batches]

    template = []
    for i in range(len(questions)):
        template.append([
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images[i],
                        },
                        {"type": "text", "text": questions[i]},
                    ],
                }
            ])
    text = processor.apply_chat_template(
                template, tokenize=False, add_generation_prompt=True
            )
    image_inputs, video_inputs = process_vision_info(template)
    input = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda')
    return input, gts, indices
    

class VizDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, get_image_example=False):
        
        self.ds_path = ds_path
        self.get_image_example = get_image_example

        self.data, self.example_dict = self._parse_data()
        self.prompt, self.example_image_list = self._get_prompt()

    def _parse_data(self):
        anno_file = os.path.join(self.ds_path, "train_annotation.json")
        data = []
        if self.get_image_example:
            example_dict = {}
        with open(anno_file, 'r') as f:
            lines = json.load(f)
            try:
                num_samples = min(len(lines), 500)
                selected_lines = random.sample(lines, num_samples)
                for line in selected_lines: # random select 500 samples
                    if '.gif' in line['imageurl']:
                        filename = str(line['imageId']) + '.gif'
                    else:
                        filename = str(line['imageId']) + '.jpg'
                    if not os.path.exists(f"{self.ds_path}/{filename}"):
                        continue
                    file_attr = list(json.loads(line['annotation'])['file_attributes'].values())[0]
                    if isinstance(file_attr, dict):
                        answer = list(file_attr.keys())[0]
                    elif isinstance(file_attr, str):
                        answer = file_attr
                    if answer == "":
                        continue
                    if self.get_image_example:
                        if answer not in example_dict and answer != "":
                            example_dict[answer] = f"{self.ds_path}/{filename}"

                    new_line = {
                        'index': line['imageId'],
                        'image': f"{self.ds_path}/{filename}",
                        'answer': answer
                    }
                    data.append(new_line)         
            except Exception as es:
                print('!!!标签文件{}解析异常: {}'.format(anno_file, es))
        print("Total num: ", len(data))

        if self.get_image_example:
            return data, example_dict
        else:
            return data, None
    
    def _get_prompt(self):
        anno_tree = json.load(open(os.path.join(self.ds_path, "anno.txt"), "r"))
        print("anno_tree: ", anno_tree)
        logger.info(f'anno tree {anno_tree}')
        anno_key = list(anno_tree.keys())[0]
        task_comment = ""
        if "task_comment" in anno_tree:
            task_comment = anno_tree["task_comment"]
        anno_list = ""
        comment_list = ""
        example_image_list = []
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

        if self.get_image_example:
            example_comments = "输出样例：\n"
            for anno, image_path in self.example_dict.items():
                example_comments += f"图<|image|>的输出标签是：{anno}\n"
                example_image_list.append(image_path)

            prompt += f"\n{example_comments}"
            
        prompt = f"USER: {prompt} ASSISTANT: "
        print("prompt: ", prompt)
        return prompt, example_image_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        index = data['index']
        image = data['image']
        answer = data['answer']
        
        return {
            'index': index,
            'images': image,
            'questions': self.prompt,
            'answers': answer,
        }


def get_dataloader(data_root, ds_name, processor):
    dataset_path = os.path.join(data_root, ds_name)
    
    random.seed(0)
    dataset = VizDataset(
        ds_path=dataset_path,
        get_image_example=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(collate_fn, processor = processor),
    )

    return dataloader



