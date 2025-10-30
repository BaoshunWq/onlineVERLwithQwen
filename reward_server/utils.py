from dataclasses import dataclass
import numpy as np
import tqdm
# from libero.libero import benchmark
import os
# from generate_intru_smolvlm import build_red_team_generator, CLIPEmbeddingModel
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import torch
import gc
from transformers import pipeline

from openvla.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from openvla.experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from openvla.experiments.robot.openvla_utils import get_processor
from openvla.experiments.robot.robot_utils import get_model



from types import SimpleNamespace

class NamespaceMapping(SimpleNamespace):
    """同时支持属性访问和字典式访问的轻量容器（递归不变性由 ns() 负责）"""
    # --- 映射接口 ---
    def __getitem__(self, k): return getattr(self, k)
    def __setitem__(self, k, v): setattr(self, k, v)
    def __delitem__(self, k): delattr(self, k)
    def get(self, k, default=None): return getattr(self, k, default)
    def __contains__(self, k): return hasattr(self, k)
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def values(self): return self.__dict__.values()
    def __iter__(self): return iter(self.__dict__)
    def __len__(self): return len(self.__dict__)

    # 辅助：转回纯 dict（便于 json 序列化/日志/做 cache key）
    def to_dict(self):
        def unwrap(x):
            if isinstance(x, NamespaceMapping):  # 递归展开
                return {k: unwrap(v) for k, v in x.__dict__.items()}
            if isinstance(x, list):
                return [unwrap(i) for i in x]
            return x
        return unwrap(self)

def ns(obj):
    """把 dict（含嵌套）递归转换为 NamespaceMapping；list 会递归处理元素。"""
    if isinstance(obj, NamespaceMapping):
        return obj
    if isinstance(obj, dict):
        return NamespaceMapping(**{k: ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ns(v) for v in obj]
    return obj



def eval_libero(cfg,model,processor,env,initial_states,task_description,is_img_perturb=False,img_perturb_func=None) -> None:

    print(f" \n----------VLA model infer in libero environment---------- ==============with annotation : {task_description} =================\n")
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # import pdb; pdb.set_trace()

    
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"


    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict[cfg.task_suite_name]()
    # num_tasks_in_suite = task_suite.n_tasks
    # task = task_suite.get_task(task_id)
    # initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO task suite

    # print(f"Task suite: {cfg.task_suite_name}")
    # log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in range(cfg.num_trials_per_task):
        # print(f"\nTask: {task_description}")
        # log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        # Setup
        t = 0
        replay_images = []

        if cfg.task_suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif cfg.task_suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif cfg.task_suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif cfg.task_suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif cfg.task_suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps

        # print(f"Starting episode {task_episodes+1}...")
        # log_file.write(f"Starting episode {task_episodes+1}...\n")
        while t < max_steps + cfg.num_steps_wait:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # Get preprocessed image
                img = get_libero_image(obs, resize_size)

                if is_img_perturb and img_perturb_func is not None:
                    img = img_perturb_func(img,method=cfg.perturb_image_method, severity=cfg.perturb_image_severity)

                # Save preprocessed image for replay video
                if cfg.is_save_video:
                    replay_images.append(img)

                # Prepare observations dict
                # Note: OpenVLA does not take proprio state as input
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                # Query model to get action
                action = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                )

                # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                action = normalize_gripper_action(action, binarize=True)

                # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    # total_successes += 1
                    break
                t += 1

            except Exception as e:
                print(f"Caught exception: {e}")
                # log_file.write(f"Caught exception: {e}\n")
                break

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode

        if cfg.is_save_video and replay_images:

            save_rollout_video(cfg.save_folder,replay_images, total_episodes, success=done, task_description=task_description)

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {total_episodes}")
        # print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # log_file.write(f"Success: {done}\n")
        # log_file.write(f"# episodes completed so far: {total_episodes}\n")
        # # log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
        # log_file.flush()
        


    # Log final results
    print(f"\nCurrent task success rate: {float(task_successes) / float(task_episodes)}")
    # print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    # log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
    # # log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
    # log_file.flush()
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
    #             f"num_episodes/{task_description}": task_episodes,
    #         }
    #     )

    # Save local log file
    # log_file.close()


    # env.close()
    # del env
    # torch.cuda.synchronize()
    # gc.collect(); torch.cuda.empty_cache()

    # print(f" \n----------libero environment closed---------- \n")

    return (task_episodes, task_successes)