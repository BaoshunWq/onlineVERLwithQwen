# reward_libero.py
# -*- coding: utf-8 -*-

# from typing import Any, Dict, List, Tuple, Optional
# from collections import OrderedDict
# import json
# import numpy as np
# import os, sys
# import requests
# import orjson

# def numpy_to_py(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.generic,)):       # np.float32/64, np.int64, np.bool_, etc.
#         return obj.item()
#     if isinstance(obj, dict):
#         return {k: numpy_to_py(v) for k, v in obj.items()}
#     if isinstance(obj, (list, tuple)):
#         return [numpy_to_py(v) for v in obj]
#     return obj


# def compute_score(batch: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
#     """
#     - 输入：batch = List[dict]，每个 item 至少包含 "response"
#              你当前每条 item 里放了 ndarray 版的 "task_suite"/"task_id"，这里做标量化处理
#     - 输出：List[{"overall": float}]（可附加更多指标键）
#     """
#     num_trials_per_task = 1

#     URL = "http://127.0.0.1:45678/score"
#     print(batch)



#     payload = {
#         "responses": [{"outputs":[{"text":numpy_to_py(item["response"])}]} for item in batch ],
#         "metas": [
#             {
#                 "original_instruction": numpy_to_py(item["ground_truth"]),
#                 "suite": numpy_to_py(item["task_suite"][0]),
#                 "task_id": numpy_to_py(item["task_id"])[0],
#                 "seed": 1
#             } for item in batch
#         ],
#         "reward_function_kwargs": {
#             "alpha": 1.0,
#             "beta": 0.1,
#             "gamma": 0.5,
#             "num_trials_per_task": num_trials_per_task,
#             "libero_cfg": {
#                 "model_family": "openvla",
#             }
#         }
#     }
    
#     # import pdb
#     # pdb.set_trace()

#     resp = requests.post(URL, json=payload, timeout=1800)
#     data = resp.json()

#     result_list = data['done_result']
#     print(resp.status_code)
#     print(resp.json())



#     return result_list


from typing import Any, Dict, List
import json, numbers, requests, numpy as np

URL = "http://127.0.0.1:45678/score"  # 按你的实际端口改

def numpy_to_py(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, dict):       return {k: numpy_to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [numpy_to_py(v) for v in o]
    return o

def pick_scalar(x):
    """把可能是 list/ndarray 的标量取出成纯 Python 标量"""
    x = numpy_to_py(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[0]
    return x

def format_reward(text: str) -> float:
    """示例：非常简单的格式得分。你可替换成更严格的校验逻辑。"""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    # 例：长度不过长、无 NaN 字样、可再加 JSON/关键词校验等
    return 1.0

def compute_score(batch: List[Dict[str, Any]], format_weight: float = 0.5, **kwargs) -> List[Dict[str, float]]:
    # 1) 构造 payload（确保全是可 JSON 序列化的类型）
    responses = [{"outputs": [{"text": str(numpy_to_py(item["response"]))}]} for item in batch]
    metas = [{
        "original_instruction": str(numpy_to_py(item.get("ground_truth", ""))),
        "suite": str(pick_scalar(item.get("task_suite", "libero_object"))),
        "task_id": int(pick_scalar(item.get("task_id", 0))),
        "seed": int(kwargs.get("seed", 1)),
    } for item in batch]

    reward_function_kwargs = {
        "alpha": 1.0, "beta": 0.1, "gamma": 0.5,
        "num_trials_per_task": int(kwargs.get("num_trials_per_task", 1)),
        "libero_cfg": {"model_family": "openvla"},
    }

    payload = {
        "responses": responses,
        "metas": metas,
        "reward_function_kwargs": reward_function_kwargs,
    }

    # 可选：本地先试编 JSON，提前发现 NaN/不可序列化
    json.dumps(payload, allow_nan=False)

    # 2) 调用远端评分（把代理关掉/忽略环境代理，避免误走代理）
    sess = requests.Session()
    sess.trust_env = False
    resp = sess.post(URL, json=payload, timeout=1800, headers={"Accept": "application/json"})
    resp.raise_for_status()
    data = resp.json()

    # 3) 取回的 accuracy 列表（你当前服务返回的是一维数字列表）
    acc_list = data.get("done_result")
    if not isinstance(acc_list, list) or len(acc_list) != len(batch):
        raise RuntimeError(f"Unexpected reward server response: {data}")

    # 4) 本地计算 format，并拼出 overall/format/accuracy
    out: List[Dict[str, float]] = []
    for item, acc in zip(batch, acc_list):
        if not isinstance(acc, numbers.Number):
            raise TypeError(f"accuracy element must be a number, got {type(acc)}")
        acc = float(acc)
        fmt = float(format_reward(str(numpy_to_py(item["response"]))))
        w = max(0.0, min(1.0, float(format_weight)))  # clamp 到 [0,1]
        overall = (1.0 - w) * acc + w * fmt
        out.append({"overall": overall, "format": fmt, "accuracy": acc})
    return out
