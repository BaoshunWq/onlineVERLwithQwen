# reward_libero.py
from typing import Any, Dict, List, Optional
import json, os, sys, math
import numpy as np

# 你的路径
sys.path.insert(0, "/home/baoshuntong/code/vlaSpace/attackVLA/LIBERO")

from dataclasses import dataclass

from LIBERO.libero.libero import benchmark
from openvla.experiments.robot.libero.libero_utils import get_libero_env
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model

# -------------- OpenVLA on LIBERO 适配器 --------------
class OpenVLAOnLIBERO:
    """
    adapter.run_episode(spec) -> {"success": 0/1, "steps_taken": int}
    必要字段：
      spec["suite"], spec["task_id"], spec["instruction_candidate"]
    可选：spec["seed"], spec["init_state_id"], spec["center_crop"]
    """
    def __init__(self, cfg: Dict[str, Any], model=None, env=None, processor=None):
        self.cfg = cfg
        self.center_crop = bool(cfg.get("center_crop", True))

        if env is None:
            suite_name = cfg["task_suite_name"]
            task_id = int(cfg["task_id"])
            benchmark_dict = benchmark.get_benchmark_dict()
            task_suite = benchmark_dict[suite_name]()
            task = task_suite.get_task(task_id)
            env, _ = get_libero_env(task, cfg.get("model_family", "openvla"), resolution=256)

        if model is None:
            model = get_model(cfg)
        if processor is None:
            processor = get_processor(cfg)

        self.vla_model = model
        self.vla_env = env
        self.processor = processor

    def run_episode(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_one_episode_real(spec)

    def _run_one_episode_real(self, spec: Dict[str, Any]) -> Dict[str, Any]:
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

        # 真实评测
        task_eps, task_succ, max_steps = eval_libero(
            self.cfg, self.vla_model, self.processor, self.vla_env,
            chosen_states, instruction, is_img_perturb=is_img_perturb, img_perturb_func=img_perturb_func
        )

        success = 1 if int(task_succ) >= 0.5 else 0
        steps_taken = int(max_steps)
        return {"success": success, "steps_taken": steps_taken}


from sentence_transformers import SentenceTransformer
_SIM_ENCODER = SentenceTransformer("BAAI/bge-m3", device="cpu")


def _sim(i0: str, ihat: str) -> float:
    if _SIM_ENCODER is None:
        # Fallback: token overlap
        a, b = set(i0.lower().split()), set(ihat.lower().split())
        if not a or not b: return 0.0
        return len(a & b) / max(1, (len(a) + len(b)) / 2)
    embs = _SIM_ENCODER.encode([i0, ihat], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))

def _length_penalty(i0: str, ihat: str) -> float:
    base = max(1, len(i0))
    return max(0.0, (len(ihat) - len(i0)) / base)

# 进程内缓存 adapter，避免反复创建 env/model
_OPENVLA_CACHE: Dict[str, OpenVLAOnLIBERO] = {}

def _adapter_key(cfg: Dict[str, Any]) -> str:
    # 只用会影响构建的大字段做 key，避免种子变化导致重建
    keys = ["task_suite_name", "task_id", "model_family", "checkpoint", "policy", "obs_mode"]
    flat = {k: cfg.get(k, None) for k in keys}
    return json.dumps(flat, sort_keys=True)

def _get_or_create_adapter(cfg: Dict[str, Any]) -> OpenVLAOnLIBERO:
    key = _adapter_key(cfg)
    if key not in _OPENVLA_CACHE:
        _OPENVLA_CACHE[key] = OpenVLAOnLIBERO(cfg=cfg)
    return _OPENVLA_CACHE[key]

# --------- 兼容 VERL 的批处理入口（必须） ----------
def compute_score(
    prompts: List[Dict[str, Any]],
    responses: List[str],
    metas: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> List[float]:
    """
    VERL/VTool-R1 的 reward_type=llm_batch 会调这个接口。
    - prompts[i]: 你喂给 Actor 的输入（我们约定含 original_instruction / suite / task_id / seed 等）
    - responses[i]: Actor 生成的“改写后指令”
    - metas[i]: 可选；如数据管线把答案或附加字段放这里，也能读到
    - kwargs: 从 yaml 的 reward_function_kwargs 传入默认 cfg/权重
    """
    # --- 超参（可在 yaml 里配） ---
    alpha = float(kwargs.get("alpha", 1.0))   # 失败奖励
    beta  = float(kwargs.get("beta", 0.1))    # 长度惩罚
    gamma = float(kwargs.get("gamma", 0.5))   # 语义相似奖励
    n_trials = int(kwargs.get("n_trials", 1))
    center_crop = bool(kwargs.get("center_crop", 1))

    # 默认 cfg（会被样本字段覆盖）
    default_cfg: Dict[str, Any] = kwargs.get("libero_cfg", {}) or {}
    default_cfg.setdefault("task_suite_name", "libero_spatial")
    default_cfg.setdefault("task_id", 0)
    default_cfg.setdefault("model_family", "openvla")
    default_cfg.setdefault("center_crop", center_crop)

    rewards: List[float] = []

    for i, (p, r) in enumerate(zip(prompts, responses)):
        # 读取样本内字段（支持放在 prompt 或 meta）
        m = metas[i] if metas and i < len(metas) else {}
        suite = (p.get("suite") or m.get("suite") or default_cfg["task_suite_name"])
        task_id = int(p.get("task_id") or m.get("task_id") or default_cfg["task_id"])
        seed = p.get("seed", m.get("seed", 0))
        init_state_id = p.get("init_state_id", m.get("init_state_id", seed))
        i0 = p.get("original_instruction") or p.get("instruction_original") or m.get("original_instruction", "")

        # 组装 cfg 并拿 adapter
        cfg = dict(default_cfg)
        cfg["task_suite_name"] = suite
        cfg["task_id"] = task_id
        adapter = _get_or_create_adapter(cfg)

        # 多次试验（可设 n_trials>1 提升稳定性）
        succ = 0
        for k in range(n_trials):
            spec = {
                "suite": suite,
                "task_id": task_id,
                "instruction_original": i0,
                "instruction_candidate": r,
                "seed": (seed + k) if isinstance(seed, int) else k,
                "init_state_id": init_state_id,
                "center_crop": center_crop,
            }
            out = adapter.run_episode(spec)
            succ += int(out.get("success", 0))
        sr = succ / max(1, n_trials)

        # 奖励：失败越多越高 + 语义保持 + 长度惩罚
        reward = alpha * (1.0 - sr) + gamma * _sim(i0, r) - beta * _length_penalty(i0, r)
        rewards.append(float(np.clip(reward, -1.0, 1.0)))

    return rewards
