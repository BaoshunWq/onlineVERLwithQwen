# ... 省略 import 与 parse_task_and_links ...
import os, json, argparse, random
from typing import List, Dict, Optional
from datasets import Dataset
from tqdm import tqdm

# 复现性
random.seed(42)

PROMPT_SYSTEM = (
    "You are a quality assurance engineer for a robot. "
    "Your goal is to come up with instructions that correctly describe the given task, "
    "are similar to what human users would give, and yet challenge the robot's ability "
    "to accomplish the task."
)

def parse_task_and_links(task_suite_name, task_to_links):
    task_suite = task_to_links[task_suite_name]
    task_language_list = list(task_suite.keys())
    return task_language_list, task_suite

def build_rlhf_messages(task: str, num_instructions: int, prefer_prompt: str = ""):
    pp = (prefer_prompt + " ") if prefer_prompt else ""
    return [
        {"role": "system", "content": PROMPT_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": "<image-0>"},   # 由 dataloader/mm_data 映射到实际路径
            {"type": "text",
             "text": f"The attached image is an example image of the initial state of a robot that will perform the task: {task}. "
                     f"{pp}Generate a diverse set of exactly {num_instructions} instructions."}
        ]},
    ]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--links_json", default="libero-init-frames/json_data_for_rl/vlm_initial_state_links_new.json")
    ap.add_argument("--out_dir", default="libero-init-frames/parquet_data_for_rl")
    ap.add_argument("--train_split", type=float, default=0.9)
    ap.add_argument("--num_instructions", type=int, default=1)  # 我们的改写任务一次只要1条原始指令
    ap.add_argument("--prefer_prompt", type=str, default="")
    # 新增：可选把 task_name 一并存下来
    ap.add_argument("--add_task_name", action="store_true")
    args = ap.parse_args()

    with open(args.links_json, "r") as f:
        task_to_links = json.load(f)

    task_language_list, task_links_suite = parse_task_and_links(args.suite, task_to_links)
    rows = []

    for idx, task_language in tqdm(enumerate(task_language_list), desc="Tasks"):
        # 1) 构造“原始指令”（可换成你更贴近人类风格的模板）
        task_text = task_language.replace("_", " ")
        original_instruction = f"Please {task_text}."

        # 2) 取一张示例初始图
        current_task_links = task_links_suite[task_language]
        if not current_task_links:
            continue
        image_path = random.choice(current_task_links)

        prompt_obj = {
            "original_instruction": original_instruction,
            "suite": args.suite,
            # 注意：这里的 task_id 先用 enumerate 的 idx；如果你有精确的 LIBERO 映射，就替换为真实 ID
            "task_id": idx,
            "seed": 0,
            "init_state_id": 0,
        }
        if args.add_task_name:
            prompt_obj["task_name"] = task_language  # 备选：在 reward 里可用 name->id 解析

        rows.append({
            "data_source": f"libero/{args.suite}",
            "prompt": prompt_obj,               # ✅ 换成我们训练要的结构
            "image": [image_path],  # ✅ 改名 image_key=images
            "answer": "",                       # ✅ 改名 answer_key=answer（占位）
        })

    assert rows, "没有产生任何样本，请检查 --links_json 与 --suite。"

    random.shuffle(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    n = len(rows); k = max(1, int(n * args.train_split))
    train_rows = rows[:k]
    val_rows = rows[k:] if k < n else rows[:1]

    Dataset.from_list(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    Dataset.from_list(val_rows).to_parquet(os.path.join(args.out_dir, "test.parquet"))
    print(f"[make_data_mm] wrote {len(train_rows)} train / {len(val_rows)} test to {args.out_dir}")



if __name__ == "__main__":
    main()