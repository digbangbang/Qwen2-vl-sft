# Qwen2-vl inference template copyed from https://github.com/QwenLM/Qwen2-VL#:~:text=Using%20%F0%9F%A4%97%20Transformers%20to%20Chat
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# a test checkpoint after sft for 2000 steps
MODEL_CKPT = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lizhiwei27/codes/models/Qwen_sft_full3/checkpoint-2000'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_CKPT,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_CKPT)

caption = '在软包的收缝上可以使用实木线条使原来得布料的撑起了,外表上更加圆滑,当然为了在触感上有较好的体验,还有在分缝处,填充条状的海绵等,如下图所示:'
image = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/meijunhui/multimodal_datasets/zero250m/375/7/1a8c447faa471fa932a2a24e2fd1f310.jpg'
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": f"现在有一张图片和对图片的描述，图片描述中可能有错误信息或与图片无关的信息。caption：“<{caption}>”。\n请参考这个caption生成新的图片描述。\n要求：\n（1）不要直接将给的caption内容拼接到生成的描述中，也不要在描述中提到词语“描述”。\n（2）不描述想象的内容，而只描述人们可以从图像中自信地确定的内容。\n（3）直接给出最终生成的描述，不要体现推理过程，不要出现“因为”等词。不以“这张图片展示了/显示了”类似的词语开头。\n（4）描述尽量详细，包括图中所有物体的属性、特征、物体之间的关系，清晰的文字信息，图片的整体氛围、场景、事件等，不低于200字。"},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)