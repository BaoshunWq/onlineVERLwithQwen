
# import torch, sys, importlib.util, os
# print("torch:", torch.__version__)
# print("torch.cuda.is_available:", torch.cuda.is_available())
# print("torch.version.cuda:", torch.version.cuda)
# print("torch built with cuda:", torch.backends.cuda.is_built())

# # 定位二进制实际路径，避免混装
# print("torch file:", torch.__file__)
# try:
#     import flash_attn
#     print("flash_attn file:", flash_attn.__file__)
# except Exception as e:
#     print("flash_attn import error:", e)

# # 检查是否混用了 conda/pip 的 torch
# import pkgutil
# has_contrib = pkgutil.find_loader('torch') is not None
# print("torch found via pkgutil:", has_contrib)

# # 打印 GPU 信息
# if torch.cuda.is_available():
#     print("device:", torch.cuda.get_device_name(0))
#     print("capability:", torch.cuda.get_device_capability(0))


# import ray

# ray.init(ignore_reinit_error=True)

# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
# print("special_tokens_map:", tok.special_tokens_map)
# print("additional_special_tokens:", tok.additional_special_tokens)

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # 必须是 VL 版本
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype="auto", device_map="auto"
)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

msgs = [
    {"role": "user", "content": [
        {"type": "image", "image": "<image-0>"},
        {"type": "text",  "text": "HELLO"}
    ]}
]
tmpl = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
print("===== CHAT TEMPLATE =====")
print(repr(tmpl))
