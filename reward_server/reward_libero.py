# reward_server/reward_libero.py
from typing import Any, Dict, List, Optional
import json, sys
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 强制 CPU 运行
# === 你的路径（按需修改）===
# 确保能 import 到 LIBERO / openvla / utils.eval_libero
sys.path.insert(0, "/root/code/saftyVLA/libero_reward_server")
sys.path.insert(0, "/root/code/saftyVLA/libero_reward_server/LIBERO")


from libero.libero import benchmark
from openvla.experiments.robot.libero.libero_utils import get_libero_env
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model
from reward_server.utils import eval_libero,ns


# ========== 默认 cfg，可被 metas/kwargs 覆盖 ==========
DEFAULT_LIBERO_CFG: Dict[str, Any] = {
    "task_suite_name": "libero_spatial",
    "task_id": 0,
    "model_family": "openvla",
    "center_crop": True,
    # "checkpoint": 自动根据 suite 选择，无需手填（也可在 libres_cfg 里显式覆盖）
    # "policy": "...",
    # "obs_mode": "rgb",
}

# ========== 根据 suite 选择对口 OpenVLA 权重 ==========
def choose_openvla_ckpt(task_suite_name: str) -> str:
    table = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object":  "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal":    "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10":      "openvla/openvla-7b-finetuned-libero-10",
    }
    if task_suite_name not in table:
        raise ValueError(f"Unknown task_suite_name: {task_suite_name}")
    return table[task_suite_name]


# ---------------- OpenVLA on LIBERO 适配器 ----------------
class OpenVLAOnLIBERO:
    """
    adapter.run_episode(spec) -> {"success": 0/1, "steps_taken": int}
    必要字段：
      spec["suite"], spec["task_id"], spec["instruction_candidate"]
    可选：spec["seed"], spec["init_state_id"], spec["center_crop"]
    """
    def __init__(self, cfg, model=None, env=None, processor=None):
        self.cfg = ns(cfg)  # 之后即可 self.cfg.task_suite_name / self.cfg["task_suite_name"] 二选一
        self.center_crop = bool(cfg.get("center_crop", True))

        # 这些对象由上层缓存提供，若未提供则兜底创建（通常不会走到）
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
            # 仅 openvla 需要 processor（保险判断）
            if cfg.get("model_family", "openvla") == "openvla":
                processor = get_processor(cfg)

        self.vla_model = model
        self.vla_env = env
        self.processor = processor

    def run_episode(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_one_episode_real(spec)

    def _run_one_episode_real(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        suite_name = spec.get("suite", "libero_object")
        task_id = int(spec.get("task_id", 0))
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if initial_states is None:
            raise RuntimeError(f"No initial states for suite={suite_name} task_id={task_id}")

        instruction = spec.get("instruction_candidate", "")
        if not instruction:
            raise ValueError("spec.instruction_candidate is required.")

        # 真实评测（你当前版本要求把全量 initial_states 传入）
        task_episodes, task_successes= eval_libero(
            self.cfg, self.vla_model, self.processor, self.vla_env,
            initial_states, instruction,
        )

        # success = 1 if int(task_successes) >= 0.5 else 0
        steps_taken = int(task_episodes)
        return {"success": task_successes, "steps_taken": steps_taken}


# ---------------- 相似度与长度惩罚 ----------------

# from sentence_transformers import SentenceTransformer
# _SIM_ENCODER = SentenceTransformer("BAAI/bge-m3", device="cpu")


# def _sim(i0: str, ihat: str) -> float:
#     if not i0 or not ihat:
#         return 0.0
#     if _SIM_ENCODER is None:
#         a, b = set(i0.lower().split()), set(ihat.lower().split())
#         if not a or not b: return 0.0
#         return len(a & b) / max(1, (len(a) + len(b)) / 2)
#     embs = _SIM_ENCODER.encode([i0, ihat], normalize_embeddings=True)
#     return float(np.dot(embs[0], embs[1]))

def _length_penalty(i0: str, ihat: str) -> float:
    base = max(1, len(i0))
    return max(0.0, (len(ihat) - len(i0)) / base)


# ---------------- vLLM 输出解析 ----------------
def _extract_text_from_vllm(resp: Any) -> str:
    """
    兼容结构：
      - "string"
      - {"text": "..."}
      - {"generated_text": "..."}
      - {"output_text": "..."}
      - {"outputs":[{"text":"..."}]}
      - {"choices":[{"text":"..."}]} 或 {"choices":[{"message":{"content":"..."}}]}
      - {"response": {...}} 递归
    """
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, list) and resp:
        return _extract_text_from_vllm(resp[0])
    if isinstance(resp, dict):
        for k in ["text", "generated_text", "output_text", "completion_text"]:
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        outputs = resp.get("outputs")
        if isinstance(outputs, list) and outputs:
            t = _extract_text_from_vllm(outputs[0])
            if t: return t
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                if isinstance(c0.get("text"), str) and c0["text"].strip():
                    return c0["text"].strip()
                msg = c0.get("message") or {}
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    return msg["content"].strip()
        inner = resp.get("response")
        if isinstance(inner, (dict, list, str)):
            t = _extract_text_from_vllm(inner)
            if t: return t
    return ""


def _get_field(*cands, default=None):
    for x in cands:
        if x is not None and x != "":
            return x
    return default


# ---------------- 共享缓存：模型/processor 与 环境 ----------------
# 模型 & processor 复用：key = (model_family, checkpoint)
_MODEL_PROC_CACHE: Dict[str, Dict[str, Any]] = {}
# 环境复用：key = (suite, task_id, model_family)
_ENV_CACHE: Dict[str, Any] = {}
# Adapter 复用（把上面两者组合进来）
_ADAPTER_CACHE: Dict[str, OpenVLAOnLIBERO] = {}

def _as_plain_dict(x):
    # 兼容 dict / NamespaceMapping
    return x.to_dict() if hasattr(x, "to_dict") else x

def _adapter_key(cfg):
    d = _as_plain_dict(cfg)
    keys = ["task_suite_name", "task_id", "model_family", "checkpoint", "policy", "obs_mode"]
    flat = {k: d.get(k, None) for k in keys}
    return json.dumps(flat, sort_keys=True)

def _model_key(cfg):
    d = _as_plain_dict(cfg)
    return json.dumps({
        "model_family": d.get("model_family", "openvla"),
        "checkpoint":   d.get("checkpoint", ""),
    }, sort_keys=True)

def _env_key(cfg):
    d = _as_plain_dict(cfg)
    return json.dumps({
        "suite":        d.get("task_suite_name"),
        "task_id":      int(d.get("task_id", 0)),
        "model_family": d.get("model_family", "openvla"),
        "resolution":   256,
    }, sort_keys=True)


def _get_or_create_model_and_processor(cfg: Dict[str, Any]):
    key = _model_key(cfg)
    mp = _MODEL_PROC_CACHE.get(key)
    if mp is not None:
        return mp["model"], mp["processor"]
    # 创建
    model = get_model(cfg)
    processor = None
    if cfg.get("model_family", "openvla") == "openvla":
        processor = get_processor(cfg)
    _MODEL_PROC_CACHE[key] = {"model": model, "processor": processor}
    return model, processor

def _get_or_create_env(cfg: Dict[str, Any]):
    key = _env_key(cfg)
    env = _ENV_CACHE.get(key)
    if env is not None:
        return env
    # 创建
    suite_name = cfg["task_suite_name"]
    task_id = int(cfg["task_id"])
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    env, _ = get_libero_env(task, cfg.get("model_family", "openvla"), resolution=256)
    _ENV_CACHE[key] = env
    return env

def _get_or_create_adapter(cfg) -> OpenVLAOnLIBERO:
    key = _adapter_key(cfg)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    # 共享对象
    model, processor = _get_or_create_model_and_processor(cfg)
    env = _get_or_create_env(cfg)
    adapter = OpenVLAOnLIBERO(cfg=cfg, model=model, env=env, processor=processor)
    _ADAPTER_CACHE[key] = adapter
    return adapter


# ---------------- 只有 responses 的批处理入口 ----------------
def compute_score(
    responses: List[Any],                 # 仅传 vLLM 的输出
    metas: Optional[List[Dict[str, Any]]] = None,  # 把任务与原指令放这里
    **kwargs
) -> List[float]:
    """
    仅接收 responses（vLLM 原始对象）与可选 metas：
    - responses[i]: str/dict/list，函数会自动抽取 candidate 指令文本
    - metas[i]: { "suite" / "task_suite_name" / "suite_name": ..., "task_id": ...,
                  "seed": ..., "init_state_id": ..., "original_instruction": ... }
    - kwargs: reward_function_kwargs（alpha/beta/gamma/n_trials/center_crop/libero_cfg）
    """

    # --- 超参 ---
    alpha = float(kwargs.get("alpha", 1.0))
    beta  = float(kwargs.get("beta", 0.1))
    gamma = float(kwargs.get("gamma", 0.5))
    n_trials = int(kwargs.get("num_trials_per_task", 1))
    center_crop = bool(kwargs.get("center_crop", 1))
    load_in_8bit = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit = False                       # (For OpenVLA only) Load with 4-bit quantization


    # 默认 cfg（可被 libero_cfg 覆盖）
    merged_cfg = dict(DEFAULT_LIBERO_CFG)
    merged_cfg.update(kwargs.get("libero_cfg", {}) or {})
    merged_cfg["center_crop"] = center_crop
    merged_cfg["load_in_8bit"] = load_in_8bit
    merged_cfg["load_in_4bit"] = load_in_4bit
    merged_cfg["seed"] = 1
    merged_cfg["num_trials_per_task"] = n_trials
    merged_cfg["num_steps_wait"] = 10
    merged_cfg["is_save_video"] = False
    merged_cfg["save_folder"] = "/tmp/libero_eval_videos"

    # 对齐 metas 长度
    metas = metas or [{}] * len(responses)
    if len(metas) < len(responses):
        metas = list(metas) + [{} for _ in range(len(responses) - len(metas))]

    succcess_list: List[float] = []

    for r, m in zip(responses, metas):

        # curr_annot_succ_list = []
        m = m or {}

        # 解析 suite / task_id
        suite = m.get("suite")
        task_id = m.get("task_id")

        # import pdb; pdb.set_trace()

        # 自动选择对口 ckpt（若未显式给 checkpoint）
        cfg = dict(merged_cfg)
        cfg["task_suite_name"] = suite
        cfg["task_id"] = task_id
        cfg["model_family"] =  "openvla"
        if not cfg.get("pretrained_checkpoint"):
            cfg["pretrained_checkpoint"] = choose_openvla_ckpt(suite)

        # 其它元信息
        seed = _get_field(m.get("seed"), 0)
        init_state_id = _get_field(m.get("init_state_id"), seed)
        i0 = _get_field(m.get("original_instruction"), "")

        # 抽 candidate 指令
        r_text = _extract_text_from_vllm(r)
        if not isinstance(r_text, str):
            r_text = str(r_text or "")
        
        cfg = ns(cfg)                 # ← 加这一行


        # 构建/复用 adapter（共享 model/processor/env）
        adapter = _get_or_create_adapter(cfg)

        spec = {
            "suite": suite,
            "task_id": task_id,
            "instruction_original": i0,
            "instruction_candidate": r_text,
            "seed": seed ,
            "init_state_id": init_state_id,
            "center_crop": center_crop,
        }
        out = adapter.run_episode(spec)
        sr = out.get("success")
        # curr_annot_succ_list.append(sr)

        # 奖励：失败越多越高 + 语义保持 + 长度惩罚
        # reward = alpha * (1.0 - sr) + gamma * _sim(i0, r_text) - beta * _length_penalty(i0, r_text)
        succcess_list.append(sr)

    return succcess_list


# if __name__ == "__main__":
#     """
#     纯本地 smoke test：
#       例1（纯字符串 response）：
#         python reward_server/reward_libero.py \
#           --suite libero_spatial --task_id 0 \
#           --response "place the red bowl onto the left shelf" \
#           --original_instruction "put the red bowl on the left shelf"

#       例2（vLLM 风格 response）：
#         python reward_server/reward_libero.py \
#           --suite libero_object --task_id 3 \
#           --use_vllm_format \
#           --response "place the red bowl onto the left shelf" \
#           --original_instruction "put the red bowl on the left shelf"

#       例3（batch 测试 + 多次试验）：
#         python reward_server/reward_libero.py \
#           --suite libero_goal --task_id 1 \
#           --n 2 --n_trials 2 --use_vLLM_format \
#           --response "move the mug into the microwave" \
#           --original_instruction "open the microwave and place the mug inside"

#       备注：
#         - 如需强制使用某 ckpt，可加：--checkpoint /abs/path/or/hf-id
#         - 不传 checkpoint 时，会按 suite 自动选择对口权重（choose_openvla_ckpt）
#     """
#     import argparse, json

#     parser = argparse.ArgumentParser(description="Smoke-test compute_score(responses, metas)")
#     parser.add_argument("--suite", default="libero_spatial",
#                         choices=["libero_spatial","libero_object","libero_goal","libero_10"])
#     parser.add_argument("--task_id", type=int, default=0)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--response", type=str,
#                         default="place the red bowl onto the left shelf")
#     parser.add_argument("--original_instruction", type=str,
#                         default="put the red bowl on the left shelf")

#     parser.add_argument("--n", type=int, default=1, help="batch size")
#     parser.add_argument("--n_trials", type=int, default=1)
#     parser.add_argument("--center_crop", type=int, default=1)

#     parser.add_argument("--alpha", type=float, default=1.0)
#     parser.add_argument("--beta", type=float, default=0.1)
#     parser.add_argument("--gamma", type=float, default=0.5)

#     parser.add_argument("--checkpoint", type=str, default="",
#                         help="可为空；为空则按 suite 自动选择")
#     parser.add_argument("--use_vllm_format", action="store_true",
#                         help="将 responses 伪装成 vLLM 风格结构进行解析测试")

#     args = parser.parse_args()

#     # 构造 responses
#     if args.use_vllm_format:
#         # vLLM 常见结构（示例之一）
#         responses = [{"outputs": [{"text": args.response}]} for _ in range(args.n)]
#     else:
#         # 纯字符串
#         responses = [args.response for _ in range(args.n)]

#     # 构造 metas（只需要与 responses 对齐；任务字段放这里）
#     metas = [{
#         "suite": args.suite,
#         "task_id": args.task_id,
#         "seed": args.seed + i,
#         "original_instruction": args.original_instruction,
#     } for i in range(args.n)]

#     # 评测超参与 libreo cfg（若不给 checkpoint，则 compute_score 内部会按 suite 自动选）
#     kwargs = {
#         "alpha": args.alpha,
#         "beta": args.beta,
#         "gamma": args.gamma,
#         "num_trials_per_task": args.n_trials,
#         "center_crop": args.center_crop,
#         "libero_cfg": {
#             "model_family": "openvla",
#         },
#     }
#     if args.checkpoint:
#         kwargs["libero_cfg"]["checkpoint"] = args.checkpoint

#     print(f"[main] suite={args.suite} task_id={args.task_id} n={args.n} n_trials={args.n_trials}")
#     rewards = compute_score(responses, metas, **kwargs)
#     print(json.dumps({"rewards": rewards}, indent=2))
