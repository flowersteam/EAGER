#!/usr/bin/env python3

"""
Generate a set of agent demonstrations.

The agent can either be a trained model or the heuristic expert (bot).

Demonstration generation can take a long time, but it can be parallelized
if you have a cluster at your disposal. Provide a script that launches
make_agent_demos.py at your cluster as --job-script and the number of jobs as --jobs.


"""

import argparse
import numpy

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
import torch.nn as nn
import matplotlib.pyplot as plt

import babyai.utils as utils

from torch.nn.utils.rnn import pad_sequence

from PIL import Image
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import FullyObsImgDirWrapper, FullyObsImgEgoWrapper, RGBImgPartialObsWrapper
from vocab import Vocab
from enum import IntEnum

from babyai.QA_simple import Model
from attrdict import AttrDict

nlp = spacy.load('en_core_web_sm')
softmax = nn.Softmax(dim=1)

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
parser.add_argument("--name-env-short", type=str, default=None,
                    help="Short name for env")
parser.add_argument("--test-QA-env", action="store_true", default=False,
                    help="Short name for env")
args = parser.parse_args()
print(args)
logger = logging.getLogger(__name__)

# Set seed for all randomness sources

stop_words = {'a', 'the', 'next', 'to', 'up', 'put', 'pick', 'after', 'then', 'and', 'open', 'you', ',', '-pron-'}
pad = 0


def numericalize(vocab, words, train=False):
    '''
    converts words to unique integers
    '''
    if not train:
        if isinstance(words, str):
            new_words = set([words]) - set(vocab.counts.keys())
        else:
            new_words = set(words) - set(vocab.counts.keys())
        if new_words:
            # replace unknown words with <<pad>>
            words = [w if w not in new_words else '<<pad>>' for w in words]
    return vocab.word2index(words, train=train)


def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()


def QG(mission, questionable_words):
    len_mission = len(mission)
    qg_data = {'questions': [], 'answers': []}
    for idx_w in range(len_mission):
        if mission[idx_w] in questionable_words:
            qg_data['questions'].append(mission[:idx_w] + ['<<question>>'] + mission[idx_w + 1:])
            qg_data['answers'].append(mission[idx_w])

    return qg_data


def generate_batch(demo):
    batch_demo = {}
    for k in demo:
        if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
            if k == 'questions':
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    len_q = len(demo[k][i])
                    for j in range(len_q):
                        batch_demo[k].append(demo[k][i][j])
            elif k == 'answers':
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    len_q = len(demo[k][i])
                    for j in range(len_q):
                        batch_demo[k].append(demo[k][i][j])

            else:
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    len_q = len(demo['questions'][i])
                    for j in range(len_q):
                        if k == 'actions':
                            batch_demo[k].append(demo[k][i])
                        elif k == 'frames':
                            frames_tensor = torch.from_numpy(demo[k][i])
                            batch_demo[k].append(frames_tensor)
                        else:
                            batch_demo[k].append(torch.unsqueeze(torch.unsqueeze(demo[k][i], 0), 0))

            if k == 'length_frames':
                batch_demo[k] = torch.cat(batch_demo[k], 0).cuda()
                batch_demo['length_frames_max'] = max(batch_demo[k]).cuda()
                # batch_demo['length_frames_max'] = max(batch_demo[k])
            elif k == 'answers':
                batch_demo[k] = torch.tensor(batch_demo[k]).cuda()

        elif k != 'length_frames_max':
            batch_demo[k] = demo[k]

    # pad  and tensorize questions
    batch_demo['questions'] = pad_sequence(
        [torch.tensor(x, dtype=torch.float32) for x in batch_demo['questions']],
        batch_first=True,
        padding_value=0).type(torch.IntTensor).cuda()

    # pad and tensorize actions
    batch_demo['actions'] = pad_sequence(
        [torch.tensor(x, dtype=torch.float32) for x in batch_demo['actions']],
        batch_first=True,
        padding_value=0).type(torch.IntTensor).cuda()

    batch_demo['frames'] = pad_sequence(batch_demo['frames'], batch_first=True,
                                        padding_value=0).cuda()

    assert batch_demo['questions'].shape[0] == batch_demo['answers'].shape[0]
    assert batch_demo['questions'].shape[0] == batch_demo['actions'].shape[0]
    assert batch_demo['questions'].shape[0] == batch_demo['frames'].shape[0]

    return batch_demo


def test_QG_QA(n_episodes, valid, model, vocabulary, seed, prob_rand_action, cumulative, cumulative_scaled,
               adjust_reward=True, confidence_reward=True,
               shift=0, gen_no_answer_question=True,
               test_env=None):
    utils.seed(seed)
    questionable_words = vocabulary['answer'].to_dict()['index2word']
    print(questionable_words)
    # Generate environment
    if args.test_QA_env:
        env = gym.make(test_env)
    else:
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

    counter_demos = 0
    checkpoint_time = time.time()

    just_crashed = False

    set_of_missions = set()
    # table of the QG_QA reward at the end of an episode
    table_reward = []
    # table of the step when each question is answered
    # table_answer_step = [[], [], [], []]
    # table of the certainty when answering an answer
    table_confidence = []
    table_frequency_no_answer = []
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
        if "BabyAI" in env.spec.id:
            agent.on_reset()

        actions = []
        mission = obs["mission"]
        print(" ")
        print(mission)
        set_of_missions.add(mission)
        images = []
        directions = []

        obss = []

        demos = {'env_ids': env.unwrapped.spec.id,
                 'missions': mission,
                 'questions': [],
                 'answers': [],
                 'frames': [],
                 'length_frames': [],
                 'length_frames_max': None,
                 'actions': []}

        # create question, answer pairs
        mission = nlp(mission)
        mission = [preprocess_token(token) for token in mission if
                   (not token.is_punct and not token.like_num)]
        qg_data = QG(mission, questionable_words)
        # numericalized
        questions = [numericalize(vocabulary['question'], x) for x in qg_data['questions']]
        answers = [numericalize(vocabulary['answer'], x) for x in qg_data['answers']]
        demos['questions'].append(questions)
        demos['answers'].append(answers)

        trajectory_reward = 0
        nbr_no_answer = 0
        step = 0
        if cumulative and cumulative_scaled:
            scale = np.zeros(len(demos['answers'][0]))
        if not cumulative:
            no_more_question = False
        # print('trajectory {}'.format(counter_demos))
        # print(mission)
        try:
            action_done = None
            while not done:
                step += 1
                if "BabyAI" in env.spec.id:

                    if np.random.rand() < prob_rand_action:
                        action = -1
                        while action in {-1, 5, 6}:
                            action = env.action_space.sample()
                        actions.append(action + 1)
                        agent.act(obs, action_done)
                    else:
                        action = agent.act(obs, action_done)['action']
                        actions.append(action.value + 1)
                    if isinstance(action, torch.Tensor):
                        action = action.item()

                    if cumulative:
                        # Visual input for the QA
                        im = obs['image']
                        # reordering to obtain images under the format CxHxW
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                        images_array = np.array(images)

                        demos['frames'].clear()
                        demos['length_frames'] = []
                        demos['actions'].clear()

                        demos['frames'].append(images_array)
                        demos['length_frames'].append(len(images_array))
                        demos['length_frames'] = torch.tensor(demos['length_frames'])
                        demos['actions'].append(actions)

                        demo_batch = generate_batch(demos)

                        with torch.no_grad():
                            answer_pred = model.forward(vocabulary['question'], **demo_batch)

                        good_answer = (torch.argmax(answer_pred['answers'], dim=1)
                                       == demo_batch['answers']).cpu().detach().numpy()
                        """for i in range(len(good_answer)):
                            if good_answer[i]:
                                table_answer_step[i].append(step)"""
                        #  evaluate confidence
                        argm = torch.argmax(answer_pred['answers'], dim=1).unsqueeze(dim=1)
                        confidence = softmax(answer_pred['answers']).gather(1, argm).squeeze(dim=1)
                        table_confidence.append(confidence.mean().cpu().detach().numpy())

                        # evaluate nbr answer no_answer

                        nbr_no_answer += (torch.argmax(answer_pred['answers'], dim=1)
                                          == torch.zeros(4).cuda()).cpu().detach().numpy()

                        if cumulative_scaled:

                            """scale[scale >= 1] += 1
                            scale[(good_answer == True) & (scale < 1)] += 1"""
                            scale[(good_answer == True)] += 1
                            inverse_squared = np.reciprocal(np.square(scale),
                                                            where=scale > 0.,
                                                            out=np.zeros(scale.shape))
                            success_pred_batch = (good_answer * inverse_squared).sum()
                            trajectory_reward += success_pred_batch
                        else:
                            success_pred_batch = good_answer.sum()
                            trajectory_reward += success_pred_batch
                    elif confidence_reward:
                        # Visual input for the QA
                        im = obs['image']
                        # reordering to obtain images under the format CxHxW
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                        images_array = np.array(images)

                        demos['frames'].clear()
                        demos['length_frames'] = []
                        demos['actions'].clear()

                        demos['frames'].append(images_array)
                        demos['length_frames'].append(len(images_array))
                        demos['length_frames'] = torch.tensor(demos['length_frames'])
                        demos['actions'].append(actions)

                        demo_batch = generate_batch(demos)

                        with torch.no_grad():
                            answer_pred = model.forward(vocabulary['question'], **demo_batch)

                        good_answer = (torch.argmax(answer_pred['answers'], dim=1)
                                       == demo_batch['answers']).cpu().detach().numpy()
                        """for i in range(len(good_answer)):
                            if good_answer[i]:
                                table_answer_step[i].append(step)"""
                        #  evaluate confidence in the right answer
                        confidence = softmax(answer_pred['answers']).gather(1, torch.unsqueeze(demo_batch['answers'], 1)).squeeze(dim=1)

                        success_pred_batch = confidence.sum().cpu().detach().numpy()
                        trajectory_reward += success_pred_batch
                    else:
                        if not no_more_question:
                            # Visual input for the QA
                            im = obs['image']
                            # reordering to obtain images under the format CxHxW
                            images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                            images_array = np.array(images)

                            demos['frames'].clear()
                            demos['length_frames'] = []
                            demos['actions'].clear()

                            demos['frames'].append(images_array)
                            demos['length_frames'].append(len(images_array))
                            demos['length_frames'] = torch.tensor(demos['length_frames'])
                            demos['actions'].append(actions)

                            demo_batch = generate_batch(demos)

                            with torch.no_grad():
                                answer_pred = model.forward(vocabulary['question'], **demo_batch)
                            not_answered_question = (torch.argmax(answer_pred['answers'], dim=1)
                                                     != demo_batch['answers']).nonzero(as_tuple=True)[
                                0].cpu().detach().numpy()
                            # Take unanswered questions to ask them anew
                            demos['questions'] = [[demos['questions'][0][i] for i in not_answered_question]]
                            demos['answers'] = [[demos['answers'][0][i] for i in not_answered_question]]

                            if adjust_reward:
                                success_pred_batch = (torch.argmax(answer_pred['answers'], dim=1)
                                                      == demo_batch['answers'])
                                argm = torch.argmax(answer_pred['answers'], dim=1)
                                len_argm = len(argm)
                                print(argm)
                                print(len_argm)
                                argm_u = torch.argmax(answer_pred['answers'], dim=1).unsqueeze(dim=1)
                                confidence = softmax(answer_pred['answers']).gather(1, argm_u).squeeze(dim=1)
                                trajectory_reward += (success_pred_batch * confidence).sum().cpu().detach().numpy()
                                table_confidence.append(confidence.mean().cpu().detach().numpy())
                                # evaluate nbr answer no_answer
                                nbr_no_answer += (argm == torch.zeros(len_argm).cuda()).sum().cpu().detach().numpy()
                            else:
                                success_pred_batch = (torch.argmax(answer_pred['answers'], dim=1)
                                                      == demo_batch['answers']).sum().cpu().detach().numpy()
                                trajectory_reward += success_pred_batch
                            if len(demos['answers'][0]) == 0:
                                no_more_question = True

                    new_obs, reward, done, _ = env.step(action)
                    agent.analyze_feedback(reward, done)

                    '''if args.include_direction:
                        directions.append(obs['direction'])'''
                else:
                    action = env.expert.act()
                    new_obs, reward, done, _ = env.step(action=action)
                    actions.append(action[0].value + 1)
                    obss.append(obs)

                obs = new_obs
                action_done = action

            if done and args.include_goal:
                if "BabyAI" in env.spec.id:
                    actions.append(MiniGridEnv.Actions.done.value + 1)
                    step += 1
                    if cumulative:
                        # Visual input for the QA
                        im = obs['image']
                        # reordering to obtain images under the format CxHxW
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))

                        images_array = np.array(images)

                        demos['frames'].clear()
                        demos['length_frames'] = []
                        demos['actions'].clear()

                        demos['frames'].append(images_array)
                        demos['length_frames'].append(len(images_array))
                        demos['length_frames'] = torch.tensor(demos['length_frames'])
                        demos['actions'].append(actions)

                        demo_batch = generate_batch(demos)
                        with torch.no_grad():
                            answer_pred = model.forward(vocabulary['question'], **demo_batch)

                        #  evaluate confidence
                        argm = torch.argmax(answer_pred['answers'], dim=1).unsqueeze(dim=1)
                        confidence = softmax(answer_pred['answers']).gather(1, argm).squeeze(dim=1)
                        table_confidence.append(confidence.mean().cpu().detach().numpy())

                        # evaluate nbr answer no_answer

                        nbr_no_answer += (torch.argmax(answer_pred['answers'], dim=1)
                                          == torch.zeros(4).cuda()).cpu().detach().numpy()
                        table_frequency_no_answer.append(nbr_no_answer / (4 * step))

                        if cumulative_scaled:
                            good_answer = (torch.argmax(answer_pred['answers'], dim=1)
                                           == demo_batch['answers']).cpu().detach().numpy()
                            """scale[scale >= 1] += 1
                            scale[(good_answer == True) & (scale < 1)] += 1"""
                            scale[(good_answer == True)] += 1
                            inverse_squared = np.reciprocal(np.square(scale).astype(int), where=scale != 0)
                            success_pred_batch = (good_answer * inverse_squared).sum()
                            trajectory_reward += success_pred_batch
                        else:
                            success_pred_batch = (torch.argmax(answer_pred['answers'], dim=1)
                                                  == demo_batch['answers']).sum().cpu().detach().numpy()
                            trajectory_reward += success_pred_batch
                    elif confidence_reward:
                        # Visual input for the QA
                        im = obs['image']
                        # reordering to obtain images under the format CxHxW
                        images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))

                        images_array = np.array(images)

                        demos['frames'].clear()
                        demos['length_frames'] = []
                        demos['actions'].clear()

                        demos['frames'].append(images_array)
                        demos['length_frames'].append(len(images_array))
                        demos['length_frames'] = torch.tensor(demos['length_frames'])
                        demos['actions'].append(actions)

                        demo_batch = generate_batch(demos)
                        with torch.no_grad():
                            answer_pred = model.forward(vocabulary['question'], **demo_batch)

                        #  evaluate confidence in the right answer
                        confidence = softmax(answer_pred['answers']).gather(1, torch.unsqueeze(demo_batch['answers'], 1)).squeeze(dim=1)

                        success_pred_batch = confidence.sum().cpu().detach().numpy()
                        trajectory_reward += success_pred_batch
                    else:
                        if not no_more_question:
                            # Visual input for the QA
                            im = obs['image']
                            # reordering to obtain images under the format CxHxW
                            images.append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))

                            images_array = np.array(images)

                            demos['frames'].clear()
                            demos['length_frames'] = []
                            demos['actions'].clear()

                            demos['frames'].append(images_array)
                            demos['length_frames'].append(len(images_array))
                            demos['length_frames'] = torch.tensor(demos['length_frames'])
                            demos['actions'].append(actions)

                            demo_batch = generate_batch(demos)
                            with torch.no_grad():
                                answer_pred = model.forward(vocabulary['question'], **demo_batch)

                            if adjust_reward:
                                success_pred_batch = (torch.argmax(answer_pred['answers'], dim=1)
                                                      == demo_batch['answers'])
                                argm = torch.argmax(answer_pred['answers'], dim=1)
                                len_argm = len(argm)
                                argm_u = torch.argmax(answer_pred['answers'], dim=1).unsqueeze(dim=1)
                                confidence = softmax(answer_pred['answers']).gather(1, argm_u).squeeze(dim=1)
                                trajectory_reward += (success_pred_batch * confidence).sum().cpu().detach().numpy()
                                table_confidence.append(confidence.mean().cpu().detach().numpy())
                                # evaluate nbr answer no_answer
                                nbr_no_answer += (argm == torch.zeros(len_argm).cuda()).sum().cpu().detach().numpy()
                            else:
                                success_pred_batch = (torch.argmax(answer_pred['answers'], dim=1)
                                                      == demo_batch['answers']).sum().cpu().detach().numpy()
                                trajectory_reward += success_pred_batch
                else:
                    obss.append(obs)

            if args.filter_steps == 0 or len(images) <= args.filter_steps:
                just_crashed = False
                if cumulative or confidence_reward:
                    table_reward.append(trajectory_reward/step)
                else:
                    table_reward.append(trajectory_reward)
                counter_demos += 1

            """if reward == 0:
                if args.on_exception == 'crash':
                    raise Exception("mission failed, the seed is {}".format(seed + counter_demos))
                just_crashed = True
                logger.info("mission failed")"""

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
            np_reward = np.array(table_reward)
            logger.info('Demo reward: {:.3f}+-{:.3f}'.format(np.mean(np_reward), np.std(np_reward)))
            checkpoint_time = now

    np_reward = np.array(table_reward)
    mean = np.mean(np_reward)
    variance = np.std(np_reward)
    logger.info('Demo reward: {:.3f}+-{:.3f}'.format(mean, variance))

    """i_step = 0
    for x in table_answer_step:
        nx = np.array(x)
        mean_step = np.mean(nx)
        variance_step = np.std(nx)
        logger.info('The question {} is correctly answered for first time at step : {:.3f}+-{:.3f}'.format(i_step, mean_step, variance_step))
        n, bins, patches = plt.hist(nx, int(1+3.322*np.log(len(nx))), density=True, facecolor='g', alpha=0.75)
        plt.grid(True)
        plt.title('Step for the first correct answer at question {}'.format(i_step))
        plt.savefig('storage/QG_QA_evaluation/hist/{}_Question_{}_prob_{}_valid.png'.format(args.name_env_short,
                                                                            i_step,
                                                                            prob_rand_action))
        plt.close()
        i_step += 1"""

    np_confidence = np.array(table_confidence)
    mean_confidence = np.mean(np_confidence)
    variance_confidence = np.std(np_confidence)

    np_no_answer = np.array(table_frequency_no_answer)
    mean_frequency_no_answer = np.mean(np_no_answer)
    variance_frequency_no_answer = np.std(np_no_answer)

    """logger.info('Confidence in the answer : {:.3f}+-{:.3f}'.format(mean_confidence, variance_confidence))"""
    return mean, variance, np.array([mean_confidence, variance_confidence]), np.array(
        [mean_frequency_no_answer, variance_frequency_no_answer])


logging.basicConfig(level='INFO', format="%(asctime)s: %(levelname)s: %(message)s")
logger.info(args)


# Test
def load_model(env, model, epoch):
    # Load voc
    demo_voc = utils.get_demos_QG_voc_path('{}_agent_done'.format(env),
                                           '{}_agent_done_QG_no_answer_vocab'.format(env), None,
                                           valid=False)
    demo_voc = demo_voc.replace('QG', 'QG_no_answer')
    vocab = utils.load_voc(demo_voc)
    # values for the model
    emb_size = len(vocab['question'])
    numb_action = 8

    attr = AttrDict()
    # TRANSFORMER settings
    # size of transformer embeddings
    attr['demb'] = 768
    # number of heads in multi-head attention
    attr['encoder_heads'] = 12
    # number of layers in transformer encoder
    attr['encoder_layers'] = 2
    # how many previous actions to use as input
    attr['num_input_actions'] = 1
    # which encoder to use for language encoder (by default no encoder)
    attr['encoder_lang'] = {
        'shared': True,
        'layers': 2,
        'pos_enc': True,
        'instr_enc': False,
    }
    # which decoder to use for the speaker model
    attr['decoder_lang'] = {
        'layers': 2,
        'heads': 12,
        'demb': 768,
        'dropout': 0.1,
        'pos_enc': True,
    }

    attr['detach_lang_emb'] = False

    # DROPOUT
    attr['dropout'] = {
        # dropout rate for language (goal + instr)
        'lang': 0.0,
        # dropout rate for Resnet feats
        'vis': 0.3,
        # dropout rate for processed lang and visual embeddings
        'emb': 0.0,
        # transformer model specific dropouts
        'transformer': {
            # dropout for transformer encoder
            'encoder': 0.1,
            # remove previous actions
            'action': 0.0,
        },
    }

    # ENCODINGS
    attr['enc'] = {
        # use positional encoding
        'pos': True,
        # use learned positional encoding
        'pos_learn': False,
        # use learned token ([WORD] or [IMG]) encoding
        'token': False,
        # dataset id learned encoding
        'dataset': False,
    }
    attr['vocab_path'] = 'storage/demos/{}_agent_done_QG_no_answer_vocab.pkl'.format(env)

    et_qa = Model(attr, emb_size, numb_action, pad=0)
    et_qa.load_state_dict(torch.load('storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(env,
                                                                                               model,
                                                                                               epoch)))
    et_qa.cuda()
    et_qa.eval()

    return et_qa, vocab


"""if args.jobs == 0:
    adjust_reward = True
    et_qa, vocab = load_model(model=10, epoch=9)
    proba = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    len_proba = len(proba)
    average_reward = np.zeros(len_proba)
    variance_reward = np.zeros(len_proba)
    for i in range(11):
        print('test of bot agent with proba {} to take a random action'.format(proba[i]))
        mean, variance = test_QG_QA(args.episodes,
                                    False,
                                    et_qa,
                                    vocab,
                                    args.seed,
                                    prob_rand_action=proba[i],
                                    adjust_reward=adjust_reward,
                                    gen_no_answer_question=False)
        average_reward[i] = mean
        variance_reward[i] = variance
    type = ''
    if adjust_reward == True:
        type = 'Adjust'
    else:
        type = 'Simple'

    with open('storage/QG_QA_evaluation/{}_{}_Reward_compare_bot_agent_rand_0_to_1.npy'.format(args.name_env_short, type), 'wb') as f:
        np.save(f, average_reward)
        np.save(f, variance_reward)

    plt.errorbar(proba, average_reward, yerr=variance_reward, fmt='-o')
    plt.title("Reward for the agent depending on the proba to take a random action")
    plt.show()

else:
    print('ISSUE not create to sustain multiple jobs for now')"""

# Validation demos
if args.valid_episodes:
    if args.test_QA_env:
        # ================= Test reward with metric "ajdusted no_answer"
        # QA trained in multienv and tested on various env ============
        # name env on which QA is trained:
        train_env = 'multienv'
        et_qa, vocab = load_model(train_env, model=0, epoch=7)

        envs = ['BabyAI-SynthThenSynthMedium-v0', 'BabyAI-Unlock-v0', 'BabyAI-Pickup-v0', 'BabyAI-PutNextMedium-v0']
        """average_reward = np.zeros(len(envs))
        variance_reward = np.zeros(len(envs))"""
        for i in range(len(envs)):
            print('test of bot agent on environment {}'.format(envs[i]))
            mean, variance, np_confidence, np_no_answer = test_QG_QA(args.valid_episodes,
                                                                     False,
                                                                     et_qa,
                                                                     vocab,
                                                                     int(1e9),
                                                                     prob_rand_action=0,
                                                                     cumulative=False,
                                                                     cumulative_scaled=False,
                                                                     adjust_reward=True,
                                                                     confidence_reward=False,
                                                                     gen_no_answer_question=False,
                                                                     test_env=envs[i])
            print(mean)
            print(variance)
            print(np_confidence)
            print(np_no_answer)
    else:
        # ================= Test different metrics with QA trained in PutNextLocal (test in the same env) ============
        # name env on which QA is trained:
        train_env = 'BabyAI-PutNextLocal-v0'
        # cumulative mean that we don't take out already answered questions
        cumulative = False
        cumulative_scaled = False
        # if not cumulative
        adjust_reward = False
        confidence_reward = True
        et_qa, vocab = load_model(train_env, model=2, epoch=9)
        proba = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

        len_proba = len(proba)
        average_reward = np.zeros(len_proba)
        variance_reward = np.zeros(len_proba)

        average_confidence = np.zeros(len_proba)
        variance_confidence = np.zeros(len_proba)

        average_freq_no_answer = np.zeros(len_proba)
        variance_freq_no_answer = np.zeros(len_proba)


        for i in range(11):
            print('test of bot agent with proba {} to take a random action'.format(proba[i]))
            mean, variance, np_confidence, np_no_answer = test_QG_QA(args.valid_episodes,
                                                                     False,
                                                                     et_qa,
                                                                     vocab,
                                                                     int(1e9),
                                                                     prob_rand_action=proba[i],
                                                                     cumulative=cumulative,
                                                                     cumulative_scaled=cumulative_scaled,
                                                                     adjust_reward=adjust_reward,
                                                                     confidence_reward=confidence_reward,
                                                                     gen_no_answer_question=False)
            average_reward[i] = mean
            variance_reward[i] = variance

            average_confidence[i] = np_confidence[0]
            variance_confidence[i] = np_confidence[1]

            average_freq_no_answer[i] = np_no_answer[0]
            variance_freq_no_answer[i] = np_no_answer[1]

            logger.info('Confidence in the answer for proba {} : {:.3f}+-{:.3f}'.format(proba[i],
                                                                                        np_confidence[0],
                                                                                        np_confidence[1]))

        type_reward = ''
        if cumulative:
            type_reward = 'cumulative'
            if cumulative_scaled:
                type_reward += '_scaled'
        elif confidence_reward:
            type_reward = 'confidence_reward'
        else:
            if adjust_reward == True:
                type_reward = 'Adjust'
            else:
                type_reward = 'Simple'

        with open(
                'storage/QG_QA_evaluation/{}_{}_Reward_compare_bot_agent_rand_0_to_1_valid.npy'.format(args.name_env_short,
                                                                                                       type_reward),
                'wb') as f:
            np.save(f, average_reward)
            np.save(f, variance_reward)

        plt.errorbar(proba, average_reward, yerr=variance_reward, fmt='-o')
        plt.title("Reward for the agent depending on the proba to take a random action")
        plt.show()
"""
    plt.errorbar(proba, average_confidence, yerr=variance_confidence, fmt='-o')
    plt.title("Confidence in the answer for the agent depending on the proba to take a random action")
    plt.show()

    plt.errorbar(proba, average_freq_no_answer, yerr=variance_freq_no_answer, fmt='-o')
    plt.title("Frequency of answer: no_answer depending on the proba to take a random action")
    plt.show()"""
"""TO LOAD 
    with open('storage/QG_QA_evaluation/test_bot_varying_randomness.npy', 'rb') as f:
        average_reward = np.load(f)
        variance_reward = np.load(f)"""
