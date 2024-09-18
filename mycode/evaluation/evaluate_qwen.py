import os
import gc
import torch

from tqdm import tqdm
from log import get_logger
from data_prepare import get_dataloader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from constant import MODEL_CKPT, PROCESSOR_PATH, DATA_ROOT

logger = get_logger()
logger.info(f'Using model ckpt {MODEL_CKPT}')

def inference(model, processor, dataloader):
    total = 0
    right_total = 0
    di_ri = {}
    di_to = {}
    for input, gts, _ in tqdm(dataloader):
        preds = model.generate(**input, max_new_tokens = 64)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input.input_ids, preds)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for gt, answer in zip(gts, output_text):
            total += 1
            right_total += int(answer == gt)

            if answer == gt:
                if answer not in di_ri.keys():
                    di_ri[answer] = 1
                else:
                    di_ri[answer] += 1
            if gt not in di_to.keys():
                di_to[gt] = 1
            else: 
                di_to[gt] += 1
        gc.collect()
        torch.cuda.empty_cache()
        # This poor code can be optimized, due to the time in internship, I just run the code frequently to finish the job.
    
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
    gc.collect()
    torch.cuda.empty_cache()

def check(ds_names):
    exist_names = []
    for name in ds_names:
        if os.path.exists(os.path.join(DATA_ROOT, name)):
            exist_names.append(name)
    return exist_names

if __name__ == '__main__':
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_CKPT,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

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
        print(ds_name)
        logger.info(f'Now test {ds_name}')
        data_loader = get_dataloader(DATA_ROOT, ds_name, processor)
        inference(model, processor, data_loader)
        