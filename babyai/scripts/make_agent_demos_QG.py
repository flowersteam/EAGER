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

nlp = spacy.load('en_core_web_sm')

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='BOT',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--demos", default=None,
                    help="path to save demonstrations (based on --model and --origin by default)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to generate demonstrations for")
parser.add_argument("--valid-episodes", type=int, default=512,
                    help="number of validation episodes to generate demonstrations for")
parser.add_argument("--seed", type=int, default=0,
                    help="start random seed")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--log-interval", type=int, default=100,
                    help="interval between progress reports")
parser.add_argument("--save-interval", type=int, default=5000,
                    help="interval between demonstrations saving")
parser.add_argument("--filter-steps", type=int, default=0,
                    help="filter out demos with number of steps more than filter-steps")
parser.add_argument("--on-exception", type=str, default='warn', choices=('warn', 'crash'),
                    help="How to handle exceptions during demo generation")

parser.add_argument("--job-script", type=str, default=None,
                    help="The script that launches make_agent_demos.py at a cluster.")
parser.add_argument("--jobs", type=int, default=0,
                    help="Split generation in that many jobs")
parser.add_argument("--full-obs", action="store_true", default=False,
                    help="Use full observations of the environment")
parser.add_argument("--ego", action="store_true", default=False,
                    help="Make full observations egocentric")
parser.add_argument("--include-goal", action="store_true", default=False,
                    help="Include an image of the final goal")
parser.add_argument("--include-direction", action="store_true", default=True,
                    help="Include list of agent orientations")
parser.add_argument("--QG-generation", action="store_true", default=False,
                    help="Make dataset for QA training")
parser.add_argument("--gen-no-answer-question", type=bool, default=False,
                    help="Generate unanswerable questions")
parser.add_argument("--biased", type=bool, default=False,
                    help="Generate biased data set for QA and for the the language classifier")

args = parser.parse_args()
logger = logging.getLogger(__name__)

# Set seed for all randomness sources

resnet = Resnet18('cpu')
# queue of other goals used to generate negative sample: no_answer-question
other_goals = deque([], maxlen=3)  # keep the last three goals used in memory
stop_words = {'a', 'the', 'next', 'to', 'up', 'put', 'pick', 'open', 'you'}

pad = 0

dict_biased_proba = {'key': {'yellow': 0.5, 'purple': 0.1, 'blue': 0.1, 'red': 0.1, 'grey': 0.1, 'green': 0.1},
                     'box': {'yellow': 0.1, 'purple': 0.5, 'blue': 0.1, 'red': 0.1, 'grey': 0.1, 'green': 0.1},
                     'ball': {'yellow': 0.1, 'purple': 0.1, 'blue': 0.5, 'red': 0.1, 'grey': 0.1, 'green': 0.1}}

def create_vocab(gen_no_answer_question):

    if gen_no_answer_question:
        v = {'question': Vocab(['<<pad>>', '<<question>>']),
             'answer': Vocab(['<<no_answer>>'])}
    else:
        v = {'question': Vocab(['<<pad>>', '<<question>>']),
             'answer': Vocab()}
    print('=V=')
    print(v)
    return v


def numericalize(vocab, words, train=True):
    '''
    converts words to unique integers
    '''
    if not train:
        if isinstance(words, str):
            words = [words]
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else '<<pad>>' for w in words]
            words = words[0]
        else:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else '<<pad>>' for w in words]
            words = words
    return vocab.word2index(words, train=train)


def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()


def QG(mission, gen_no_answer_question, seed):
    len_mission = len(mission)
    qg_data = {'questions': [], 'answers': []}
    for idx_w in range(len_mission):
        # if mission[idx_w] in questionable_words:
        if mission[idx_w] not in stop_words:
            vocab_answer_qg.add(mission[idx_w])
            list_vocab_answer.append(mission[idx_w])
            qg_data['questions'].append(mission[:idx_w] + ['<<question>>'] + mission[idx_w + 1:])
            qg_data['answers'].append(mission[idx_w])

    # generating no_answer-question
    if gen_no_answer_question and len(other_goals) == other_goals.maxlen:
        wrong_mission = other_goals[0]
        if wrong_mission != mission:
            len_wrong_mission = len(wrong_mission)
            similarities = 0
            for idx_w in range(len_wrong_mission):
                # select randomly some wrong answers to avoid having too much of them
                question = []
                answer = []
                if wrong_mission[idx_w] not in stop_words:
                    question.append(wrong_mission[:idx_w] + ['<<question>>'] + wrong_mission[idx_w + 1:])
                    answer.append('<<no_answer>>')
                    if wrong_mission[idx_w] in qg_data['answers']:
                        similarities += 1
                # we add to qg_data the questions with a proba link to the similarity of the wrong mission
                # with the true mission, missions that have a high similarity are rarer, so they are favored
                # proba_selection = [0.1, 0.1, 0.15, 0.25, 0.3]
                # p_s = proba_selection[similarities]
                # the function below has been designed to obtain similar value than the one commented above
                # that have been empirically found  
                p_s = 0.325 / (1 + np.exp(6.75 - 3 * similarities)) + 0.095
                for q, a in zip(question, answer):
                    if np.random.rand() < p_s:
                        vocab_answer_qg.add('<<no_answer>>')
                        list_vocab_answer.append('<<no_answer>>')
                        qg_data['questions'].append(q)
                        qg_data['answers'].append(a)

                        if similarities not in number_similarities.keys():
                            number_similarities[similarities] = 1
                        else:
                            number_similarities[similarities] += 1

    return qg_data


def tensorized(demos):
    # we don't pad sequence of image here
    # because it is too heavy for the memory
    # seqs_images = [torch.tensor(x) for x in demos['frames']]
    # seqs_images_numpy = pad_sequence(seqs_images, batch_first=True, padding_value=pad).numpy()
    demos_tens_pad = {'env_ids': demos['env_ids'],
                      'missions': demos['missions'],
                      'questions': demos['questions'],
                      'answers': demos['answers'],
                      'frames': demos['frames'],
                      'length_frames': torch.tensor(demos['length_frames']),
                      'length_frames_max': None,  # calculated during creation of batches  for train
                      'actions': demos['actions']}
    return demos_tens_pad


def print_demo_lengths(demos, counter_demos):
    # num_frames_per_episode = [len(demo) for demo in demos['frames']]
    num_frames_per_episode = list(map(len, demos['actions'][-counter_demos:]))
    logger.info('Demo length: {:.3f}+-{:.3f}'.format(
        np.mean(num_frames_per_episode), np.std(num_frames_per_episode)))


def generate_demos(n_episodes, valid, seed, shift=0, gen_no_answer_question=True):
    utils.seed(seed)

    # Generate environment
    env = gym.make(args.env)

    if args.full_obs:
        logger.info("using full observations")
        env = FullyObsImgDirWrapper(env)
    if args.ego:
        logger.info("using egocentric view")
        env = FullyObsImgEgoWrapper(env)
    """if not args.full_obs and not args.ego:
        logger.info("using RGB partial view")
        env = RGBImgPartialObsWrapper(env)"""

    if "BabyAI" in env.spec.id:
        agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, args.env)
    if args.QG_generation:
        demos_path = utils.get_demos_QG_path(args.demos, args.env, 'agent_done', valid)
        if gen_no_answer_question:
            demos_path = str(demos_path).replace('.pkl', '_no_answer.pkl')
    else:
        demos_path = utils.get_demos_path(args.demos, args.env, 'agent', valid)
    demos = {'env_ids': [],
             'missions': [],
             'questions': [],
             'answers': [],
             'frames': [],
             'length_frames': [],
             'actions': []}
    if args.biased and args.env == "BabyAI-PutNextLocal-v0":
        demo_l_class = {'questions': [],
                        'answers': []
                        }
    counter_demos = 0
    checkpoint_time = time.time()

    just_crashed = False

    set_of_missions = set()
    while True:
        if counter_demos == n_episodes:
            break

        done = False
        if just_crashed:
            logger.info("reset the environment to find a mission that the bot can solve")
            env.reset()
        else:
            env.seed(seed + counter_demos)
        obs = env.reset()

        if args.biased and args.env == "BabyAI-PutNextLocal-v0":
            m = nlp(obs["mission"])
            adj1 = str(m[2])
            obj1 = str(m[3])
            adj2 = str(m[7])
            obj2 = str(m[8])
            while dict_biased_proba[obj1][adj1]*dict_biased_proba[obj2][adj2]< np.random.rand():
                env.reset()
                obs = env.reset()
                m = nlp(obs["mission"])
                adj1 = str(m[2])
                obj1 = str(m[3])
                adj2 = str(m[7])
                obj2 = str(m[8])

        if "BabyAI" in env.spec.id:
            agent.on_reset()

        actions = []
        mission = obs["mission"]
        set_of_missions.add(mission)
        images = []
        directions = []

        obss = []

        try:
            action_done = None
            while not done:
                if "BabyAI" in env.spec.id:

                    if np.random.rand() < 0:
                        action = -1
                        while action in {-1, 5, 6}:
                            action = env.action_space.sample()
                        actions.append(action + 1)
                        agent.act(obs, action_done)
                    else:
                        action = agent.act(obs, action_done)['action']
                        actions.append(action.value + 1)
                    # action = agent.act(obs)['action']
                    if isinstance(action, torch.Tensor):
                        action = action.item()
                    new_obs, reward, done, _ = env.step(action)
                    agent.analyze_feedback(reward, done)
                    # actions.append(action.value + 1)
                    im = obs['image']
                    # reordering to obtain images under the format CxHxW
                    images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                    '''if args.include_direction:
                        directions.append(obs['direction'])'''
                else:
                    action = env.expert.act()
                    new_obs, reward, done, _ = env.step(action=action)
                    actions.append(action[0].value + 1)
                    obss.append(obs)
                action_done = action
                obs = new_obs
            if done and args.include_goal:
                if "BabyAI" in env.spec.id:
                    actions.append(MiniGridEnv.Actions.done.value + 1)
                    im = obs['image']
                    # reordering to obtain images under the format HxWxC
                    images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                    '''if args.include_direction:
                        directions.append(obs['direction'])'''
                else:
                    obss.append(obs)
            if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                if args.QG_generation:
                    if "BabyAI" in env.spec.id:
                        images = np.array(images)
                        mission = nlp(mission)
                        mission = [preprocess_token(token) for token in mission if
                                   (not token.is_punct and not token.like_num)]
                        # add goal to the queue of goals for no_answer-question
                        if gen_no_answer_question:
                            other_goals.append(mission)
                        # create question, answer pairs
                        qg_data = QG(mission, gen_no_answer_question, seed)
                        if args.biased and args.env == "BabyAI-PutNextLocal-v0":
                            qg_data_language_classifier = QG(mission, False, seed)

                        # numericalized
                        questions = [numericalize(vocab['question'], x) for x in qg_data['questions']]
                        answers = [numericalize(vocab['answer'], x) for x in qg_data['answers']]
                        if args.biased and args.env == "BabyAI-PutNextLocal-v0":
                            l_class_questions = [numericalize(vocab['question'], x) for x in qg_data_language_classifier['questions']]
                            l_class_answers = [numericalize(vocab['answer'], x) for x in qg_data_language_classifier['answers']]
                            demo_l_class['questions'].append(l_class_questions)
                            demo_l_class['answers'].append(l_class_answers)
                        demos['env_ids'].append(env.unwrapped.spec.id)
                        demos['missions'].append(mission)
                        demos['questions'].append(questions)
                        demos['answers'].append(answers)
                        demos['frames'].append(blosc.pack_array(images))
                        demos['length_frames'].append(len(images))
                        demos['actions'].append(actions)
                        """demos.append((env.unwrapped.spec.id, mission, blosc.pack_array(np_images_extracted), questions,
                                      answers, actions))"""
                        counter_demos += 1
                    else:
                        images = np.array(images)
                        # actions = [0, 1]
                        demos['env_ids'].append(env.unwrapped.spec.id)
                        demos['missions'].append(mission)
                        demos['questions'].append(questions)
                        demos['answers'].append(answers)
                        demos['frames'].append(blosc.pack_array(images))
                        demos['length_frames'].append(len(images))
                        demos['actions'].append(actions)
                        """ demos.append((env.unwrapped.spec.id, mission, obss, directions, actions))
                        """
                        counter_demos += 1
                else:
                    if "BabyAI" in env.spec.id:
                        images = np.array(images)
                        mission = nlp(mission)
                        mission = [preprocess_token(token) for token in mission if
                                   (not token.is_punct and not token.like_num)]
                        # add goal to the queue of goals for no_answer-question
                        if gen_no_answer_question:
                            other_goals.append(mission)
                        # create question, answer pairs
                        qg_data = QG(mission, gen_no_answer_question, seed)
                        # numericalized
                        questions = [numericalize(vocab['question'], x) for x in qg_data['questions']]
                        answers = [numericalize(vocab['answer'], x) for x in qg_data['answers']]
                        demos['env_ids'].append(env.unwrapped.spec.id)
                        demos['missions'].append(mission)
                        demos['questions'].append(questions)
                        demos['answers'].append(answers)
                        demos['frames'].append(blosc.pack_array(images))
                        demos['length_frames'].append(len(images))
                        demos['actions'].append(actions)
                        """demos.append((env.unwrapped.spec.id, mission, blosc.pack_array(np_images_extracted), questions,
                                      answers, directions, actions))"""
                        counter_demos += 1
                    else:
                        images = np.array(images)
                        demos['env_ids'].append(env.unwrapped.spec.id)
                        demos['missions'].append(mission)
                        demos['questions'].append(questions)
                        demos['answers'].append(answers)
                        demos['frames'].append(blosc.pack_array(images))
                        demos['length_frames'].append(len(images))
                        demos['actions'].append(actions)
                        """demos.append((env.unwrapped.spec.id, mission, obss, directions, actions))"""
                        counter_demos += 1
                just_crashed = False

            if reward == 0:
                if args.on_exception == 'crash':
                    raise Exception("mission failed, the seed is {}".format(seed + counter_demos))
                just_crashed = True
                logger.info("mission failed")
        except (Exception, AssertionError):
            if args.on_exception == 'crash':
                raise
            just_crashed = True
            logger.exception("error while generating demo #{}".format(counter_demos))
            # continue

        if counter_demos >= 0 and counter_demos % args.log_interval == 0:
            now = time.time()
            demos_per_second = args.log_interval / (now - checkpoint_time)
            to_go = (n_episodes - counter_demos) / demos_per_second
            logger.info("demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                counter_demos - 1, demos_per_second, to_go))
            checkpoint_time = now

        # Save demonstrations
        if args.save_interval > 0 and counter_demos < n_episodes and counter_demos % args.save_interval == 0:
            logger.info("Saving demos...")
            demo_tens = tensorized(demos)
            # utils.save_demos(demo_tens, str(demos_path).replace('.pkl', '_{}.pkl'.format(counter_demos // args.save_interval)))
            if args.biased and args.env == "BabyAI-PutNextLocal-v0":
                utils.save_demos(demo_tens, str(demos_path).replace('.pkl', '_biased.pkl'))
                utils.save_demos(demo_l_class, str(demos_path).replace('.pkl', '_biased_l_class.pkl'))
            else:
                utils.save_demos(demo_tens, demos_path)
            logger.info("{} demos saved".format(counter_demos))
            print_demo_lengths(demos, counter_demos)
            """demos = {'env_ids': [],
                     'missions': [],
                     'questions': [],
                     'answers': [],
                     'frames': [],
                     'actions': []}"""

    # Save demonstrations
    logger.info("Saving demos...")
    demo_tens = tensorized(demos)
    if args.biased and args.env == "BabyAI-PutNextLocal-v0":
        utils.save_demos(demo_tens, str(demos_path).replace('.pkl', '_biased.pkl'))
        utils.save_demos(demo_l_class, str(demos_path).replace('.pkl', '_biased_l_class.pkl'))
        with open("{}.txt".format(str(demos_path).replace('.pkl', '_biased')), "w") as output:
            output.write(str(list(set_of_missions)))
    else:
        utils.save_demos(demo_tens, demos_path)
        with open("{}.txt".format(str(demos_path).replace('.pkl', '')), "w") as output:
            output.write(str(list(set_of_missions)))
    logger.info("{} demos saved".format(counter_demos))
    if not valid:
        if args.biased and args.env == "BabyAI-PutNextLocal-v0":
            pkl.dump(vocab, open(str(demos_path).replace('.pkl', '_biased_vocab.pkl'), "wb"))
        else:
            pkl.dump(vocab, open(str(demos_path).replace('.pkl', '_vocab.pkl'), "wb"))



def generate_multi_env_demos(n_episodes, valid, seed, shift=0, gen_no_answer_question=True):
    # This function is executed if args.env = multienv

    utils.seed(seed)

    envs = ['BabyAI-Unlock-v0', 'BabyAI-Pickup-v0', 'BabyAI-PutNextLocal-v0', 'BabyAI-SynthThenSynthMedium-v0']
    n_episodes_tab = [n_episodes, n_episodes, n_episodes, n_episodes]
    len_envs = len(envs)
    demos = {'env_ids': [],
             'missions': [],
             'questions': [],
             'answers': [],
             'frames': [],
             'length_frames': [],
             'actions': []}
    demos_path = utils.get_demos_QG_path(args.demos, args.env, 'agent_done', valid)
    if gen_no_answer_question:
        demos_path = str(demos_path).replace('.pkl', '_no_answer.pkl')
    set_of_missions = set()
    for i in range(len_envs):
        logger.info("==== {} ====".format(envs[i]))
        # Generate environment
        env = gym.make(envs[i])

        if args.full_obs:
            logger.info("using full observations")
            env = FullyObsImgDirWrapper(env)
        if args.ego:
            logger.info("using egocentric view")
            env = FullyObsImgEgoWrapper(env)
        """if not args.full_obs and not args.ego:
            logger.info("using RGB partial view")
            env = RGBImgPartialObsWrapper(env)"""

        if "BabyAI" in env.spec.id:
            agent = utils.load_agent(env, args.model, args.demos, 'agent', args.argmax, envs[i])

        counter_demos = 0
        checkpoint_time = time.time()

        just_crashed = False

        while True:
            if counter_demos == n_episodes_tab[i]:
                break

            done = False
            if just_crashed:
                logger.info("reset the environment to find a mission that the bot can solve")
                env.reset()
            else:
                env.seed(seed + counter_demos)
            obs = env.reset()
            if "BabyAI" in env.spec.id:
                agent.on_reset()

            actions = []
            mission = obs["mission"]
            set_of_missions.add(mission)
            images = []
            directions = []

            obss = []
            randomness_of_the_agent = np.random.choice(np.array([0., 0.1, 0.4, 0.8]),
                                                       size=1,
                                                       p=[0.45, 0.35, 0.1, 0.1])

            try:
                action_done = None
                while not done:
                    if "BabyAI" in env.spec.id:
                        if np.random.rand() < randomness_of_the_agent:
                            action = -1
                            while action in {-1, 5, 6}:
                                action = env.action_space.sample()
                            actions.append(action + 1)
                            agent.act(obs, action_done)
                        else:
                            action = agent.act(obs, action_done)['action']
                            actions.append(action.value + 1)
                        # action = agent.act(obs)['action']
                        if isinstance(action, torch.Tensor):
                            action = action.item()
                        new_obs, reward, done, _ = env.step(action)
                        agent.analyze_feedback(reward, done)
                        # actions.append(action.value + 1)
                        im = obs['image']
                        # reordering to obtain images under the format CxHxW
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                        '''if args.include_direction:
                            directions.append(obs['direction'])'''
                    else:
                        action = env.expert.act()
                        new_obs, reward, done, _ = env.step(action=action)
                        actions.append(action[0].value + 1)
                        obss.append(obs)
                    action_done = action
                    obs = new_obs
                if done and args.include_goal:
                    if "BabyAI" in env.spec.id:
                        actions.append(MiniGridEnv.Actions.done.value + 1)
                        im = obs['image']
                        # reordering to obtain images under the format HxWxC
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                        '''if args.include_direction:
                            directions.append(obs['direction'])'''
                    else:
                        obss.append(obs)
                if reward > 0 and (args.filter_steps == 0 or len(images) <= args.filter_steps):
                    if args.QG_generation:
                        if "BabyAI" in env.spec.id:
                            images = np.array(images)
                            mission = nlp(mission)
                            mission = [preprocess_token(token) for token in mission if
                                       (not token.is_punct and not token.like_num)]
                            # add goal to the queue of goals for no_answer-question
                            if gen_no_answer_question:
                                other_goals.append(mission)
                            # create question, answer pairs
                            qg_data = QG(mission, gen_no_answer_question, seed)
                            # numericalized
                            questions = [numericalize(vocab['question'], x) for x in qg_data['questions']]
                            answers = [numericalize(vocab['answer'], x) for x in qg_data['answers']]
                            demos['env_ids'].append(env.unwrapped.spec.id)
                            demos['missions'].append(mission)
                            demos['questions'].append(questions)
                            demos['answers'].append(answers)
                            demos['frames'].append(blosc.pack_array(images))
                            demos['length_frames'].append(len(images))
                            demos['actions'].append(actions)
                            """demos.append((env.unwrapped.spec.id, mission, blosc.pack_array(np_images_extracted), questions,
                                          answers, actions))"""
                            counter_demos += 1
                        else:
                            images = np.array(images)
                            # actions = [0, 1]
                            demos['env_ids'].append(env.unwrapped.spec.id)
                            demos['missions'].append(mission)
                            demos['questions'].append(questions)
                            demos['answers'].append(answers)
                            demos['frames'].append(blosc.pack_array(images))
                            demos['length_frames'].append(len(images))
                            demos['actions'].append(actions)
                            """ demos.append((env.unwrapped.spec.id, mission, obss, directions, actions))
                            """
                            counter_demos += 1
                    else:
                        if "BabyAI" in env.spec.id:
                            images = np.array(images)
                            mission = nlp(mission)
                            mission = [preprocess_token(token) for token in mission if
                                       (not token.is_punct and not token.like_num)]
                            # add goal to the queue of goals for no_answer-question
                            if gen_no_answer_question:
                                other_goals.append(mission)
                            # create question, answer pairs
                            qg_data = QG(mission, gen_no_answer_question, seed)
                            # numericalized
                            questions = [numericalize(vocab['question'], x) for x in qg_data['questions']]
                            answers = [numericalize(vocab['answer'], x) for x in qg_data['answers']]
                            demos['env_ids'].append(env.unwrapped.spec.id)
                            demos['missions'].append(mission)
                            demos['questions'].append(questions)
                            demos['answers'].append(answers)
                            demos['frames'].append(blosc.pack_array(images))
                            demos['length_frames'].append(len(images))
                            demos['actions'].append(actions)
                            """demos.append((env.unwrapped.spec.id, mission, blosc.pack_array(np_images_extracted), questions,
                                          answers, directions, actions))"""
                            counter_demos += 1
                        else:
                            images = np.array(images)
                            demos['env_ids'].append(env.unwrapped.spec.id)
                            demos['missions'].append(mission)
                            demos['questions'].append(questions)
                            demos['answers'].append(answers)
                            demos['frames'].append(blosc.pack_array(images))
                            demos['length_frames'].append(len(images))
                            demos['actions'].append(actions)
                            """demos.append((env.unwrapped.spec.id, mission, obss, directions, actions))"""
                            counter_demos += 1
                    just_crashed = False

                if reward == 0:
                    if args.on_exception == 'crash':
                        raise Exception("mission failed, the seed is {}".format(seed + counter_demos))
                    just_crashed = True
                    logger.info("mission failed")
            except (Exception, AssertionError):
                if args.on_exception == 'crash':
                    raise
                just_crashed = True
                logger.exception("error while generating demo #{}".format(counter_demos))
                # continue

            if counter_demos >= 0 and counter_demos % args.log_interval == 0:
                now = time.time()
                demos_per_second = args.log_interval / (now - checkpoint_time)
                to_go = (n_episodes_tab[i] - counter_demos) / demos_per_second
                logger.info("demo #{}, {:.3f} demos per second, {:.3f} seconds to go".format(
                    counter_demos - 1, demos_per_second, to_go))
                checkpoint_time = now

            # Save demonstrations
            if args.save_interval > 0 and counter_demos < n_episodes_tab[i] and counter_demos % args.save_interval == 0:
                logger.info("Saving demos...")
                demo_tens = tensorized(demos)
                # utils.save_demos(demo_tens, str(demos_path).replace('.pkl', '_{}.pkl'.format(counter_demos // args.save_interval)))
                utils.save_demos(demo_tens, demos_path)
                logger.info("{} demos saved".format(counter_demos))
                print_demo_lengths(demos, counter_demos)

        print(" ")

    # Save demonstrations
    logger.info("Saving demos...")
    demo_tens = tensorized(demos)
    utils.save_demos(demo_tens, demos_path)
    logger.info("{} demos saved".format(counter_demos))
    if not valid:
        pkl.dump(vocab, open(str(demos_path).replace('.pkl', '_vocab.pkl'), "wb"))
    with open("{}.txt".format(str(demos_path).replace('.pkl', '')), "w") as output:
        output.write(str(list(set_of_missions)))


logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)
# Training demos

if args.jobs == 0:
    """demo_voc = utils.get_demos_QG_voc_path('{}_agent_done'.format('multienv2'), None, None, valid=False)
    demo_voc = demo_voc.replace('QG', 'QG_no_answer')
    print(demo_voc)
    vocab = utils.load_voc(demo_voc)
    questionable_words = vocab['answer'].counts.keys()"""
    vocab = create_vocab(gen_no_answer_question=args.gen_no_answer_question)
    # analyse the vocabulary of the question
    vocab_answer_qg = set()
    list_vocab_answer = []
    # measure of similarities between the original true mission
    # and the false mission due to the structure of PNR mission there are maximum 4 similarities
    number_similarities = dict()
    print('no_answer {}'.format(args.gen_no_answer_question))
    if "multienv" in args.env:
        generate_multi_env_demos(args.episodes, False, args.seed, gen_no_answer_question=args.gen_no_answer_question)
    else:
        generate_demos(args.episodes, False, args.seed, gen_no_answer_question=args.gen_no_answer_question)
    vocab_answer_counts = Counter(list_vocab_answer)
    df = pd.DataFrame.from_dict(vocab_answer_counts, orient='index')
    df.plot(kind='bar')
    plt.show()
    print('Similarities')
    print(number_similarities)

else:
    print('ISSUE not create to sustain multiple jobs for now')
    # generate_demos_cluster()
# Validation demos
if args.valid_episodes:
    # analyse the vocabulary of the question
    vocab_answer_qg = set()
    list_vocab_answer = []
    number_similarities = dict()
    if "multienv" in args.env:
        generate_multi_env_demos(args.valid_episodes, True, int(1e9),
                                 gen_no_answer_question=args.gen_no_answer_question)
    else:
        generate_demos(args.valid_episodes, True, int(1e9), gen_no_answer_question=args.gen_no_answer_question)
    vocab_answer_counts = Counter(list_vocab_answer)
    df = pd.DataFrame.from_dict(vocab_answer_counts, orient='index')
    df.plot(kind='bar')
    plt.show()
    print('Similarities')
    print(number_similarities)
