# reward_libero.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict
import json
import numpy as np
import os, sys
import requests
import orjson

def numpy_to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):       # np.float32/64, np.int64, np.bool_, etc.
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_py(v) for v in obj]
    return obj


def compute_score(batch: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    - 输入：batch = List[dict]，每个 item 至少包含 "response"
             你当前每条 item 里放了 ndarray 版的 "task_suite"/"task_id"，这里做标量化处理
    - 输出：List[{"overall": float}]（可附加更多指标键）
    """
    num_trials_per_task = 1

    URL = "http://127.0.0.1:34567/score"
    print(batch)



    payload = {
        "responses": [{"outputs":[{"text":numpy_to_py(item["response"])}]} for item in batch ],
        "metas": [
            {
                "original_instruction": numpy_to_py(item["ground_truth"]),
                "suite": numpy_to_py(item["task_suite"]),
                "task_id": numpy_to_py(item["task_id"]),
                "seed": 1
            } for item in batch
        ],
        "reward_function_kwargs": {
            "alpha": 1.0,
            "beta": 0.1,
            "gamma": 0.5,
            "num_trials_per_task": num_trials_per_task,
            "libero_cfg": {
                "model_family": "openvla",
            }
        }
    }
    


    resp = requests.post(URL, json=payload, timeout=1800)
    data = resp.json()
    import pdb
    pdb.set_trace()
    result_list = data['done_result']
    print(resp.status_code)
    print(resp.json())



    return result_list
