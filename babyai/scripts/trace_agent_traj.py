#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import gym
import gym_minigrid.window
import logging
import sys
import subprocess
import os
import time
import numpy as np
import blosc
import torch
import spacy
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import babyai.utils as utils

from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import FullyObsImgDirWrapper, FullyObsImgEgoWrapper, RGBImgPartialObsWrapper
from nn.enc_visual import Resnet18
from vocab import Vocab
from collections import deque
from collections import Counter

# Set seed for all randomness sources

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE_PIXELS = 32

def generate_traj(model, env, seed):


    utils.seed(seed)
    # Generate environment
    env = gym.make(env)

    for i in [4]:
        model_i = model.format(i)
        agent = torch.load('storage/models/'+model_i+'/model.pt')
        obss_preprocessor = utils.ObssPreprocessor(model_i, env.observation_space, None)
        if torch.cuda.is_available():
            agent.eval()
            agent.cuda()

        memory = torch.zeros(1, agent.memory_size, device=device)
        mask = torch.ones(1, device=device)

        done = False
        env.seed(seed)
        obs = env.reset()
        print(obs['mission'])

        action_record = []

        for i in range(100):
            env.render(mode='human', highlight=True, tile_size=TILE_PIXELS)
        while not done:

            preprocessed_obs = obss_preprocessor([obs], device=device)
            model_results = agent(preprocessed_obs, memory)

            dist = model_results['dist']
            memory = model_results['memory']
            # action = dist.sample().cpu().numpy()
            action = torch.argmax(dist.probs).cpu().numpy()
            action_record.append(int(action))
            # print("Action: {}".format(action))

            new_obs, reward, done, _ = env.step(action)


            for i in range(100):
                env.render(mode='human', highlight=True, tile_size=TILE_PIXELS)

            obs = new_obs

def generate_average_traj(model, env, seed):


    utils.seed(seed)
    # Generate environment
    env = gym.make(env)

    env.window = gym_minigrid.window.Window('gym_minigrid')
    env.window.show(block=False)

    """if args.full_obs:
        logger.info("using full observations")
        env = FullyObsImgDirWrapper(env)
    if args.ego:
        logger.info("using egocentric view")
        env = FullyObsImgEgoWrapper(env)
    if not args.full_obs and not args.ego:
        logger.info("using RGB partial view")
        env = RGBImgPartialObsWrapper(env)"""
    traj = np.zeros((env.width, env.height), dtype=np.int32)
    for i in range(1, 5):
        model_i = model.format(i)
        agent = torch.load('storage/models/'+model_i+'/model.pt')
        obss_preprocessor = utils.ObssPreprocessor(model_i, env.observation_space, None)
        if torch.cuda.is_available():
            agent.eval()
            agent.cuda()

        memory = torch.zeros(1, agent.memory_size, device=device)
        mask = torch.ones(1, device=device)

        done = False
        env.seed(seed)
        obs = env.reset()
        print(obs['mission'])


        if traj[env.agent_pos[0]][env.agent_pos[1]] is None:
            traj[env.agent_pos[0]][env.agent_pos[1]] = 1
        else:
            traj[env.agent_pos[0]][env.agent_pos[1]] += 1

        action_record = []

        # for i in range(10):
            # env.render(mode='human', highlight=True, tile_size=TILE_PIXELS)
        while not done:

            preprocessed_obs = obss_preprocessor([obs], device=device)
            model_results = agent(preprocessed_obs, memory)

            dist = model_results['dist']
            memory = model_results['memory']
            # action = dist.sample().cpu().numpy()
            action = torch.argmax(dist.probs).cpu().numpy()
            action_record.append(int(action))
            # print("Action: {}".format(action))

            new_obs, reward, done, _ = env.step(action)

            if traj[env.agent_pos[0]][env.agent_pos[1]] is None:
                traj[env.agent_pos[0]][env.agent_pos[1]] = 1
            else:
                traj[env.agent_pos[0]][env.agent_pos[1]] += 1

            #for i in range(10):
            # env.render(mode='human', highlight=True, tile_size=TILE_PIXELS)

            obs = new_obs

        print(traj)
        # plt.scatter(np.arange(len(action_record)), action_record)
        # plt.show()

    traj = traj/4
    env.seed(seed)
    _ = env.reset()
    img = env.grid.render_traj(tile_size=TILE_PIXELS, agent_pos=env.agent_pos, agent_dir=env.agent_dir, traj=traj)
    print(traj)
    for j in range(200):
        env.window.show_img(img)
        env.window.set_caption(env.mission)


generate_traj('QG_QA/PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_{}', 'BabyAI-PutNextLocal-v0', 5)
# generate_traj('QG_QA/paral_SEQ-adjusted-train_env-multienv3_no_answer-lambda_026-model-0_10-seed_{}_best', 'BabyAI-SynthThenSynthMedium-v0', 0)
# generate_traj('QG_QA/SEQ-GTM-RS-Online-05-D-{}_best', 'BabyAI-SynthThenSynthMedium-v0', 0)

"""for g in range(4, 10):
    generate_average_traj('onl/PNL-GTL-RS-Online-025-D-{}_best', 'BabyAI-PutNextLocal-v0', g)
    generate_average_traj('QG_QA/PNL-adjusted-train_env-PNL_no_answer-lambda_24-model-2_10-seed_{}', 'BabyAI-PutNextLocal-v0', g)"""

