import os
import cv2
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR

# import isaacgym
from legged_gym.envs import *
from legged_gym.gym_utils import get_args, export_policy_as_jit, task_registry, Logger

import torch
from tqdm import tqdm
from datetime import datetime

import genesis as gs


def play(args):
    gs.init(logging_level="warning")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1200  # number of steps before plotting states
    for i in tqdm(range(stop_state_log)):

        actions = policy(obs.detach())  # * 0.

        if FIX_COMMAND:
            env.commands[:, 0] = 0.5  # 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            # env.commands[:, 3] = 0.

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    FIX_COMMAND = False
    EXPORT_POLICY = False
    args = get_args()
    play(args)

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
