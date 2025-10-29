# reward_libero.py
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict
import json
import numpy as np
import os, sys

# （按你的项目路径调整；确保能 import 到 LIBERO / openvla）
sys.path.insert(0, "/root/code/saftyVLA/onlineVERLwithQwen/LIBERO")

from LIBERO.libero.libero import benchmark
from openvla.experiments.robot.libero.libero_utils import get_libero_env
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model
from types import SimpleNamespace

def _as_cfg_obj(cfg: Dict[str, Any]):
    """把 dict -> 支持属性访问的对象；已是对象就原样返回。"""
    # 若已经是对象且有 model_family 属性，直接用
    if not isinstance(cfg, dict) and hasattr(cfg, "model_family"):
        return cfg
    # dict -> SimpleNamespace（顶层键可用点号访问）
    return SimpleNamespace(**cfg)


# =========================
# 可选：语义相似（默认 gamma=0 不启用）
# =========================
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

_SIM_ENCODER = None  # 懒加载

def _get_sim_encoder():
    global _SIM_ENCODER
    if _SIM_ENCODER is None and SentenceTransformer is not None:
        # 轻量起见放 CPU；如需加速可改 device="cuda"
        _SIM_ENCODER = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return _SIM_ENCODER

def _sim(i0: str, ihat: str) -> float:
    enc = _get_sim_encoder()
    if enc is None:
        # Fallback：token overlap
        a, b = set(i0.lower().split()), set(ihat.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, (len(a) + len(b)) / 2)
    embs = enc.encode([i0, ihat], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))

def _length_penalty(i0: str, ihat: str) -> float:
    base = max(1, len(i0))
    return max(0.0, (len(ihat) - len(i0)) / base)


# =========================
# 模型/Processor 全局缓存（共享一份或按 ckpt 共享）
# =========================
_MODEL_CACHE: Dict[Tuple[Any, Any, Any], Tuple[Any, Any]] = {}  # (model, processor)

def _get_or_create_model_proc(cfg: Dict[str, Any]):
    key = (
        cfg.get("model_family", "openvla"),
        cfg.get("checkpoint", None),
        cfg.get("policy", None),
    )
    if key not in _MODEL_CACHE:
        
        model = get_model(cfg)
        processor = get_processor(cfg)
        _MODEL_CACHE[key] = (model, processor)
    return _MODEL_CACHE[key]



# =========================
# 环境 LRU 缓存（最多 maxsize 份）
# =========================
class _EnvLRU:
    def __init__(self, maxsize: int = 10):
        self.maxsize = maxsize
        self.store: "OrderedDict[Tuple[str, int, bool], Any]" = OrderedDict()

    def get(self, key):
        env = self.store.pop(key, None)
        if env is not None:
            self.store[key] = env
        return env

    def put(self, key, env):
        if key in self.store:
            self.store.pop(key)
        self.store[key] = env
        if len(self.store) > self.maxsize:
            old_key, old_env = self.store.popitem(last=False)
            # 尽量释放资源
            if hasattr(old_env, "close"):
                try:
                    old_env.close()
                except Exception:
                    pass
            # 如需强制回收 CUDA，可视情况启用：
            # import gc, torch; gc.collect(); torch.cuda.empty_cache()

_ENV_LRU = _EnvLRU(maxsize=10)

def _get_or_create_env(cfg: Dict[str, Any]):
    suite_name = cfg["task_suite_name"]
    task_id = int(cfg["task_id"])
    center_crop = bool(cfg.get("center_crop", True))
    ekey = (suite_name, task_id, center_crop)

    env = _ENV_LRU.get(ekey)
    if env is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()
        task = task_suite.get_task(task_id)
        env, _ = get_libero_env(task, cfg.get("model_family", "openvla"), resolution=256)
        _ENV_LRU.put(ekey, env)
    return env


# =========================
# OpenVLA on LIBERO 适配器（复用模型 + LRU 环境）
# =========================
class OpenVLAOnLIBERO:
    """
    adapter.run_episode(spec) -> {"success": 0/1, "steps_taken": int}
    必要字段：
      spec["suite"], spec["task_id"], spec["instruction_candidate"]
    可选：spec["seed"], spec["init_state_id"], spec["center_crop"]
    """
    def __init__(self, cfg: Dict[str, Any], model=None, env=None, processor=None):
        cfg = _as_cfg_obj(cfg)           # ← 新增：包装成可属性访问
        self.cfg = cfg
        self.center_crop = bool(cfg.get("center_crop", True))

        # 统一从缓存获取（可选实参优先）
        if model is None or processor is None:
            model, processor = _get_or_create_model_proc(cfg)
        if env is None:
            env = _get_or_create_env(cfg)

        self.vla_model = model
        self.vla_env = env
        self.processor = processor

    def run_episode(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_one_episode_real(spec)

    def _run_one_episode_real(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        # 运行一次真实评测
        from utils import eval_libero

        suite_name = spec.get("suite", "libero_object")
        task_id = int(spec.get("task_id", 0))
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if not initial_states:
            raise RuntimeError(f"No initial states for suite={suite_name} task_id={task_id}")
        init_idx = int(spec.get("init_state_id", 0)) % len(initial_states)
        chosen_states = [initial_states[init_idx]]

        instruction = spec.get("instruction_candidate", "")
        if not instruction:
            raise ValueError("spec.instruction_candidate is required.")

        is_img_perturb = bool(spec.get("is_img_perturb", False))
        img_perturb_func = spec.get("img_perturb_func", None)

        task_eps, task_succ, max_steps = eval_libero(
            self.cfg, self.vla_model, self.processor, self.vla_env,
            chosen_states, instruction, is_img_perturb=is_img_perturb, img_perturb_func=img_perturb_func
        )

        success = 1 if int(task_succ) >= 0.5 else 0
        steps_taken = int(max_steps)
        return {"success": success, "steps_taken": steps_taken}


# =========================
# 轻量的 Adapter 缓存（按任务维度分身；共享模型）
# =========================
_OPENVLA_CACHE: Dict[str, OpenVLAOnLIBERO] = {}

def _adapter_key(cfg: Dict[str, Any]) -> str:
    # 影响“需要不同 env”的字段才进 key
    keys = ["task_suite_name", "task_id", "center_crop"]
    return json.dumps({k: cfg.get(k, None) for k in keys}, sort_keys=True)

def _get_or_create_adapter(cfg: Dict[str, Any]) -> OpenVLAOnLIBERO:
    key = _adapter_key(cfg)
    if key not in _OPENVLA_CACHE:
        # 注意：__init__ 内会从缓存取【共享模型】和【LRU env】
        _OPENVLA_CACHE[key] = OpenVLAOnLIBERO(cfg=cfg)
    return _OPENVLA_CACHE[key]


def compute_score(batch: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    - 输入：batch = List[dict]，每个 item 至少包含 "response"
             你当前每条 item 里放了 ndarray 版的 "task_suite"/"task_id"，这里做标量化处理
    - 输出：List[{"overall": float}]（可附加更多指标键）
    """
    # --------- 小工具：把 ndarray / numpy 标量 / 单元素 list 转为 Python 标量 ----------
    def _to_py_scalar(x, default=None):
        try:
            import numpy as np  # 局部导入，避免上层无 numpy 时报错
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return default
                if x.ndim == 0:
                    return x.item()
                return _to_py_scalar(x.reshape(-1)[0], default)
            if isinstance(x, (np.generic,)):
                return x.item()
        except Exception:
            pass
        # 单元素 list/tuple
        if isinstance(x, (list, tuple)) and len(x) == 1:
            return _to_py_scalar(x[0], default)
        return x if x is not None else default

    # --------- 超参（最小可跑：默认不启用相似度/长度项） ----------
    alpha = float(kwargs.get("alpha", 1.0))
    beta  = float(kwargs.get("beta", 0.0))
    gamma = float(kwargs.get("gamma", 0.0))
    n_trials = int(kwargs.get("n_trials", 1))
    max_envs = int(kwargs.get("max_envs", 10))

    # 动态调整 LRU 上限（如果你在别处定义了 _ENV_LRU）
    global _ENV_LRU
    if "_ENV_LRU" in globals() and _ENV_LRU.maxsize != max_envs:
        _ENV_LRU.maxsize = max_envs

    # --------- 默认 cfg（可被每条样本覆盖） ----------
    default_suite = (
        _to_py_scalar(kwargs.get("task_suite_name"))
        or _to_py_scalar(kwargs.get("task_suite"))
        or _to_py_scalar(kwargs.get("suite"))
        or "libero_spatial"
    )
    default_task  = int(_to_py_scalar(kwargs.get("task_id"), 0))
    default_center_crop = bool(_to_py_scalar(kwargs.get("center_crop"), 1))
    default_cfg = {
        "task_suite_name": default_suite,
        "task_id": default_task,
        "model_family": _to_py_scalar(kwargs.get("model_family"), "openvla"),
        "center_crop": default_center_crop,
        "checkpoint": _to_py_scalar(kwargs.get("checkpoint"), None),
        "policy": _to_py_scalar(kwargs.get("policy"), None),
    }

    out_scores: List[Dict[str, float]] = []

    for item in batch:
        # —— 从样本里读取；容忍 ndarray/list/tuple 形式，并做标量化 —— 
        suite = (
            _to_py_scalar(item.get("suite"))
            or _to_py_scalar(item.get("task_suite"))     # 你当前传的就是这个（ndarray）
            or _to_py_scalar(item.get("data_source"))
            or default_suite
        )
        task_id = int(_to_py_scalar(item.get("task_id"), default_task))  # 你当前传的是 ndarray
        center_crop = bool(_to_py_scalar(item.get("center_crop"), default_center_crop))
        r = str(_to_py_scalar(item.get("response"), ""))                 # candidate 指令
        i0 = str(_to_py_scalar(item.get("original_instruction"), ""))    # 可空
        seed = int(_to_py_scalar(item.get("seed"), 0))
        init_state_id = int(_to_py_scalar(item.get("init_state_id"), seed))

        if not r:
            out_scores.append({"overall": 0.0})
            continue

        # 组 cfg，拿 adapter（内部会用模型缓存 + 环境 LRU）
        cfg = dict(default_cfg)
        cfg.update({"task_suite_name": str(suite), "task_id": int(task_id), "center_crop": bool(center_crop)})

        adapter = _get_or_create_adapter(cfg)

        # —— 评测 ——（可设 n_trials>1 提升稳定性）
        succ = 0
        for k in range(max(1, n_trials)):
            spec = {
                "suite": str(suite),
                "task_id": int(task_id),
                "instruction_original": i0,
                "instruction_candidate": r,
                "seed": seed + k,
                "init_state_id": init_state_id,
                "center_crop": bool(center_crop),
            }
            out = adapter.run_episode(spec)  # -> {"success": 0/1, "steps_taken": int}
            succ += int(out.get("success", 0))
        sr = succ / max(1, n_trials)

        # —— 最小奖励：失败越多越高（你要诱导失败） + 可选项 —— 
        sim = _sim(i0, r) if gamma > 0.0 else 0.0
        len_pen = _length_penalty(i0, r) if beta > 0.0 else 0.0
        reward = alpha * (1.0 - sr) + gamma * sim - beta * len_pen

        out_scores.append({
            "overall": float(np.clip(reward, -1.0, 1.0)),
            "sr": float(sr),
            "sim": float(sim),
            "len_pen": float(len_pen),
        })

    return out_scores

