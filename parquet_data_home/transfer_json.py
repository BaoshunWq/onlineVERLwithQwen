import json

# === 输入输出文件路径 ===
input_file = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"         # 原始 JSON 文件
output_file = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # 修改后的输出文件
# https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames/libero_spatial_task-1_img-0_frame0.png
# === 旧前缀与新前缀 ===
old_prefix = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames"
old_prefix_2 = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main"
new_prefix = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames"
# /libero_object_task-0_img-0_frame0.png
# === 旧路径中任务文件夹名（可根据你新数据集修改） ===
# old_folder = "libero_init_frames/"
# new_folder = "libero_object_task-0_img-0_frame0.png"  # 如果只是更换前缀，可留空或替换对应部分
# /libero_object_task-0_img-0_frame0.png
# === 读入 JSON ===
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for suite_name, suite in data.items():                 # suite: dict
    for task, img_urls in suite.items():               # img_urls: list[str]
        for i, url in enumerate(img_urls):
            if url.startswith(old_prefix):
                new_url = url.replace(old_prefix, new_prefix)
            elif url.startswith(old_prefix_2):
                new_url = url.replace(old_prefix_2, new_prefix)
                # 若还要改子路径，继续链式替换：
                # new_url = new_url.replace("libero_init_frames/", "libero_object_task-0_img-0_frame0.png")
            img_urls[i] = new_url                   # 关键：写回列表

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ 已更新 {len(data)} 条记录，结果保存在 {output_file}")

# import json

# # === 输入输出文件路径 ===
# input_file = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto.json"         # 原始 JSON 文件
# output_file = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto_new.json"  # 修改后的输出文件
# # https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames/libero_spatial_task-1_img-0_frame0.png
# # === 旧前缀与新前缀 ===
# # old_prefix = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames"
# # new_prefix = "/home/baoshuntong/code/vlaSpace/attackVLA/libero-init-frames"
# # /libero_object_task-0_img-0_frame0.png
# # === 旧路径中任务文件夹名（可根据你新数据集修改） ===
# # old_folder = "libero_init_frames/"
# # new_folder = "libero_object_task-0_img-0_frame0.png"  # 如果只是更换前缀，可留空或替换对应部分
# # /libero_object_task-0_img-0_frame0.png
# # === 读入 JSON ===
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for example in data:                 # suite: dict
#     response = example["response"].split(".")[1] 


#     example["response"] = response+"."


# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

# print(f"✅ 已更新 {len(data)} 条记录，结果保存在 {output_file}")



# import json

# # === 输入输出文件路径 ===
# input_file = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto.json"         # 原始 JSON 文件
# output_file = "/home/baoshuntong/code/saftyEmbodyAI/embodied-red-teaming/input_json_data/libero_spatial_kto_new.json"  # 修改后的输出文件
# # https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames/libero_spatial_task-1_img-0_frame0.png
# # === 旧前缀与新前缀 ===
# old_prefix = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames"
# new_prefix = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main"
# # /libero_object_task-0_img-0_frame0.png
# # === 旧路径中任务文件夹名（可根据你新数据集修改） ===
# # old_folder = "libero_init_frames/"
# # new_folder = "libero_object_task-0_img-0_frame0.png"  # 如果只是更换前缀，可留空或替换对应部分
# # /libero_object_task-0_img-0_frame0.png
# # === 读入 JSON ===
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for example in data:                 # suite: dict
#     img_url = example["image"]               # img_urls: list[str]
#     if img_url.startswith(old_prefix):
#         new_url = img_url.replace(old_prefix, new_prefix)
#         # 若还要改子路径，继续链式替换：
#         # new_url = new_url.replace("libero_init_frames/", "libero_object_task-0_img-0_frame0.png")
#         example['image'] = new_url                   # 关键：写回列表

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

# print(f"✅ 已更新 {len(data)} 条记录，结果保存在 {output_file}")


# import json

# # === 输入输出文件路径 ===
# input_file = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames/json_data_for_rl/vlm_initial_state_links.json"         # 原始 JSON 文件
# output_file = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json"  # 修改后的输出文件
# # https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames/libero_spatial_task-1_img-0_frame0.png
# # === 旧前缀与新前缀 ===
# old_prefix = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main/libero_init_frames"
# old_prefix_2 = "https://huggingface.co/datasets/TBS2001/libero-init-frames/resolve/main"
# new_prefix = "/root/code/saftyVLA/onlineVERLwithQwen/libero-init-frames"
# # /libero_object_task-0_img-0_frame0.png
# # === 旧路径中任务文件夹名（可根据你新数据集修改） ===
# # old_folder = "libero_init_frames/"
# # new_folder = "libero_object_task-0_img-0_frame0.png"  # 如果只是更换前缀，可留空或替换对应部分
# # /libero_object_task-0_img-0_frame0.png
# # === 读入 JSON ===
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)

# for example in data:                 # suite: dict
#     prompt = example["prompt"]               # img_urls: list[str]
#     if prompt.endswith("[attached]"):
#         prompt = prompt.replace("[attached]", "<image>")
#         # 若还要改子路径，继续链式替换：
#         # new_url = new_url.replace("libero_init_frames/", "libero_object_task-0_img-0_frame0.png")
#         example['prompt'] = prompt                   # 关键：写回列表

# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data, f, indent=4, ensure_ascii=False)

# print(f"✅ 已更新 {len(data)} 条记录，结果保存在 {output_file}")