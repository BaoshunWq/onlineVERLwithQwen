import os
from dataclasses import dataclass
import numpy as np
import tqdm
from libero.libero import benchmark
import wandb
from typing import List, Dict, Any
import numpy as np
from smolvlm_ppo_libero.ovla_libero_adapter import OpenVLAOnLIBERO


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
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)



def eval_libero(cfg, model,processor,env,initial_states,task_description,is_img_perturb=False,img_perturb_func=None) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    # log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        print(f"\nTask: {task_description}")
        # log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        env.reset()

        # Set initial states
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

        print(f"Starting episode {task_episodes+1}...")
        # log_file.write(f"Starting episode {task_episodes+1}...\n")
        while t < max_steps + cfg.num_steps_wait:

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



        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode

        if cfg.is_save_video:
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description)

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {total_episodes}")

    # Log final results
    print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")


    return (task_episodes, task_successes, t)



# def length_penalty(text: str, target_len: int = 64) -> float:
#     # mild quadratic penalty above target_len
#     over = max(0, len(text.split()) - target_len)
#     return (over / 10.0) ** 2

# def reward_fn(cfg,samples: List[Dict[str, Any]],
#               suite: str,
#               task_id: int,
#               n_trials: int,
#               max_steps: int,
#               alpha: float = 1.0,
#               beta: float = 0.1,
#               center_crop: int = 1,
#               # pretrained_checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial",
#               use_real: int = 0) -> Dict[str, np.ndarray]:
#     """
#     Expected sample format from VERL:
#       sample["prompt"]  -> I0 (original task description)
#       sample["response"]-> Ĩ (candidate rewritten instruction)
#     Returns:
#       {"rewards": np.array([...], dtype=np.float32)}
#     """
#     shared = SharedInit(
#         pretrained_checkpoint=cfg.pretrained_checkpoint,
#         center_crop=bool(center_crop),
#         use_real=bool(use_real),
#         verbose=False,
#     )
#     adapter = OpenVLAOnLIBERO(cfg,shared)

#     rewards = []
#     for s in samples:
#         I0 = s.get("prompt", "")
#         I_hat = s.get("response", "")
#         # Build episode specs for this instruction
#         successes = []
#         # simple deterministic seeds for n_trials
#         for k in range(n_trials):
#             spec = {
#                 "suite": suite,
#                 "task_id": int(task_id),
#                 "instruction_original": I0,
#                 "instruction_candidate": I_hat,
#                 "seed": k,
#                 "init_state_id": k,  # in real setting, you may want to sample valid IDs instead
#                 "max_steps": int(max_steps),
#                 "center_crop": bool(center_crop),
#             }
#             out = adapter.run_episode(spec)
#             successes.append(int(out.get("success", 0)))
#         sr = float(sum(successes)) / max(1, len(successes))
#         pen = length_penalty(I_hat)
#         R = alpha * (1.0 - sr) - beta * pen
#         rewards.append(R)

#     return {"rewards": np.array(rewards, dtype=np.float32)}