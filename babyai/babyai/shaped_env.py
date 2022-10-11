import gc
import gym
import torch
import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process, Pipe
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import logging
import babyai.utils as utils
import time

import spacy

nlp = spacy.load('en_core_web_sm')
stop_words = {'a', 'the', 'next', 'to', 'up', 'put', 'pick', 'after', 'then', 'and', 'open', 'you', ',', '-pron-', 'go'}
pad = 0

from babyai.QA_simple import Model
from attrdict import AttrDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def load_model(no_answer, train_env, model_QA, epoch_QA):
    '''
    loading QA model and the associated vocabulary
    '''
    # Load voc
    demo_voc = utils.get_demos_QG_voc_path('{}_agent_done'.format(train_env), train_env, None,
                                           valid=False)
    if no_answer == True:
        demo_voc = demo_voc.replace("QG_vocab.pkl", "QG_no_answer_vocab.pkl")

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
    if no_answer:
        attr['vocab_path'] = demo_voc
        et_qa = Model(attr, emb_size, numb_action, pad=0)
        et_qa.load_state_dict(torch.load('storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                                   model_QA,
                                                                                                   epoch_QA)))
    else:
        attr['vocab_path'] = demo_voc
        et_qa = Model(attr, emb_size, numb_action, pad=0)
        et_qa.load_state_dict(torch.load('storage/models/{}/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                         model_QA,
                                                                                         epoch_QA)))
    et_qa.cuda()
    et_qa.eval()

    return et_qa, vocab


def preprocess_token(token):
    '''
    Reduce token to its lowercase lemma form
    '''
    return token.lemma_.strip().lower()


def QG(mission, questionable_words):
    '''
    Question generation module
    '''
    len_mission = len(mission)
    qg_data = {'questions': [], 'answers': []}
    for idx_w in range(len_mission):
        # if mission[idx_w] not in stop_words:
        if mission[idx_w] in questionable_words:
            qg_data['questions'].append(mission[:idx_w] + ['<<question>>'] + mission[idx_w + 1:])
            qg_data['answers'].append(mission[idx_w])

    return qg_data


def numericalize(vocab, words, train=False):
    '''
    converts words to unique integers
    '''

    if not train:
        new_words = set(words) - set(vocab.counts.keys())
        if new_words:
            # replace unknown words with <<pad>>
            words = [w if w not in new_words else '<<pad>>' for w in words]
    return vocab.word2index(words, train=train)


def generate_batch(demo, bs):
    '''
    generates a batch that will be pass to the QA
    '''
    batch_demo = {}
    nbQueries = []
    for k in demo:
        if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
            if k == 'questions':
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    len_q = len(demo[k][i])
                    nbQueries.append(len_q)
                    for j in range(len_q):
                        batch_demo[k].append(torch.tensor(demo[k][i][j], dtype=torch.long))

            elif k == 'answers':
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    len_q = len(demo[k][i])
                    for j in range(len_q):
                        batch_demo[k].append(demo[k][i][j][0])

            else:
                batch_demo[k] = []
                for i in range(len(demo[k])):
                    if k == 'actions':
                        batch_demo[k].append(torch.tensor(demo[k][i], dtype=torch.long))
                    elif k == 'frames':
                        frames_tensor = torch.from_numpy(demo[k][i])
                        batch_demo[k].append(frames_tensor)
                    else:
                        batch_demo[k].append(torch.unsqueeze(torch.unsqueeze(demo[k][i], 0), 0))

            if k == 'length_frames':
                batch_demo[k] = torch.cat(batch_demo[k], 0)
                batch_demo['length_frames_max'] = max(batch_demo[k])
            elif k == 'answers':
                batch_demo[k] = torch.tensor(batch_demo[k])

        elif k != 'length_frames_max':
            batch_demo[k] = demo[k]

    # pad  and tensorize questions
    batch_demo['questions'] = pad_sequence(batch_demo['questions'],
                                           batch_first=True,
                                           padding_value=0).cuda()

    # pad and tensorize actions
    batch_demo['actions'] = pad_sequence(batch_demo['actions'],
                                         batch_first=True,
                                         padding_value=0).cuda()

    batch_demo['frames'] = pad_sequence(batch_demo['frames'],
                                        batch_first=True,
                                        padding_value=0).cuda()

    maxNb = max(nbQueries)

    nbQueries = torch.tensor(nbQueries).view(bs, 1).expand(bs, maxNb)
    validQuery = torch.tensor(np.arange(maxNb)).view(1, maxNb).expand(bs, -1)
    validQuery = torch.where(validQuery >= nbQueries, False, True).view(-1)

    frames_tensor = batch_demo['frames'].view(bs, 1, -1)
    frames_tensor = frames_tensor.expand(-1, maxNb, -1)
    frames_tensor = frames_tensor.reshape(bs * maxNb, batch_demo['length_frames_max'], 3, 7, 7)
    batch_demo['frames'] = frames_tensor[validQuery]

    actions_tensor = batch_demo['actions'].view(bs, 1, -1)
    actions_tensor = actions_tensor.expand(-1, maxNb, -1)
    actions_tensor = actions_tensor.reshape(bs * maxNb, batch_demo['length_frames_max'])
    batch_demo['actions'] = actions_tensor[validQuery]

    # batch_demo['answers'] = batch_demo['answers'].cuda()

    assert batch_demo['questions'].shape[0] == batch_demo['answers'].shape[0]
    assert batch_demo['questions'].shape[0] == batch_demo['frames'].shape[0]
    assert batch_demo['questions'].shape[0] == batch_demo['actions'].shape[0]

    return batch_demo


def multi_worker(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask)
        if cmd == "step":
            ret = []
            for env, a, stopped in zip(envs, data[0], data[1]):
                if not stopped:
                    obs, reward, done, info = env.step(a)
                    if done:
                        obs = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                ret.append(env.reset())
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            mode, highlight = data
            ret = envs[0].render(mode, highlight)
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


def multi_worker_cont(conn, envs):
    """Target for a subprocess that handles a set of envs"""
    while True:
        cmd, data = conn.recv()
        # step(actions, stop_mask)
        if cmd == "step":
            ret = []
            for env, a, stopped in zip(envs, data[0], data[1]):
                if not stopped:
                    obs, reward, done, info = env.step(action=a)
                    if done:
                        obs = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        # reset()
        elif cmd == "reset":
            ret = []
            for env in envs:
                ret.append(env.reset())
            conn.send(ret)
        # render_one()
        elif cmd == "render_one":
            mode = data
            ret = envs[0].render(mode)
            conn.send(ret)
            # __str__()
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


class ParallelShapedEnv(gym.Env):
    """Parallel environment that holds a list of environments and can
       evaluate a low-level policy for use in reward shaping.
    """

    def __init__(self,
                 envs,  # List of environments
                 pi_l=None,  # Low-level policy or termination classifier
                 done_action=None,  # Output of pi_l indicating done
                 instr_handler=None,  # InstructionHandler for low-level demos
                 reward_shaping=None,  # Reward shaping type
                 subtask_cls=None,  # Subtask relevance classifier
                 subtask_cls_preproc=None,  # Instruction preprocessor
                 subtask_online_ds=None,  # Dataset for subtask classifier
                 subtask_discount=1,  # Discount for done subtask count
                 learn_baseline_cls=None,  # LEARN baseline classifier
                 learn_baseline_preproc=None,  # LEARN baseline classifier
                 type_QG_QA_reward=None,  # QGQA type of reward simple or adjusted
                 no_answer_question=True,  # use a model with possibility to answer "no_answer"
                 train_env=None,  # name of env used for training
                 model_QA=None,  # scheme used for training QA
                 epoch_QA=None,  # epoch of the trained model used
                 stateactionpredictor=None,  # if you use a reward based on curiosity
                 obss_preprocessor=None
                 ):
        assert len(envs) >= 1, "No environment provided"
        self.envs = envs
        self.num_envs = len(self.envs)
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        self.spec = deepcopy(self.envs[0].unwrapped.spec)
        self.spec_id = f"ParallelShapedEnv<{self.spec.id}>"
        self.env_name = self.envs[0].unwrapped.spec.id
        self.action_space = self.envs[0].action_space
        self.pi_l = pi_l
        self.done_action = done_action
        self.instr_handler = instr_handler
        if self.instr_handler:
            self.num_subtasks = self.instr_handler.D_l_size()
        self.reward_shaping = reward_shaping
        self.subtask_cls = subtask_cls
        self.subtask_cls_preproc = subtask_cls_preproc
        self.subtask_online_ds = subtask_online_ds
        self.subtask_discount = float(subtask_discount)
        self.learn_baseline_cls = learn_baseline_cls
        self.learn_baseline_preproc = learn_baseline_preproc
        self.type_QG_QA_reward = type_QG_QA_reward
        self.stateactionpredictor = stateactionpredictor
        self.obss_preprocessor = obss_preprocessor

        if "BabyAI" in self.env_name:
            self.envs_per_proc = 64
        elif "BabyPANDA" in self.env_name:
            self.envs_per_proc = 1
        else:
            self.envs_per_proc = 64

        if self.reward_shaping in ["subtask_oracle_ordered"]:
            # Setup stacks to hold oracle subtasks
            self.stacks = [[] for _ in range(self.num_envs)]

        if self.reward_shaping in ["subtask_classifier_static",
                                   "subtask_classifier_online",
                                   "subtask_classifier_static_unclipped",
                                   "subtask_classifier_online_unclipped"]:
            # Setup arrays to keep track of which subtasks are completed
            # during episode, and past bonuses
            self.pi_l_already_done_relevant = np.array(
                [[False for j in range(self.num_subtasks)]
                 for i in range(self.num_envs)]
            )
            self.pi_l_already_done_all = np.array(
                [[False for j in range(self.num_subtasks)]
                 for i in range(self.num_envs)]
            )
            self.past_pi_l_done_discounted = np.array(
                [0. for i in range(self.num_envs)]
            )
            # Setup list to keep track of instructions to store for dataset
            self.tasks_instr = ["" for _ in range(self.num_envs)]
            # Setup array to record of which environments have succeeded
            # at the high-level task    
            self.tasks_succeeded = np.array([False for _ in range(self.num_envs)])

        if self.reward_shaping in ["learn_baseline"]:
            # Setup array to record unnormalized action frequencies
            assert "Discrete" in str(type(self.envs[0].action_space))
            self.num_actions = self.envs[0].action_space.n
            self.action_freqs = np.array(
                [[0 for j in range(self.num_actions)]
                 for i in range(self.num_envs)]
            )

        if self.reward_shaping in ["QG_QA"]:
            self.QA, self.vocabulary = load_model(no_answer_question,
                                                  train_env,
                                                  model_QA=model_QA,
                                                  epoch_QA=epoch_QA)
            self.questionable_words = self.vocabulary['answer'].to_dict()['index2word']

            self.questions = [[] for _ in range(self.num_envs)]
            self.questions_len_begin = np.zeros(self.num_envs)
            self.questions_answered_len_current = np.zeros(self.num_envs)
            self.frames = [[] for _ in range(self.num_envs)]
            self.actions = [[] for _ in range(self.num_envs)]

            self.answers = [[] for _ in range(self.num_envs)]

            self.rewards_bonus = np.zeros(self.num_envs)

            self.rewards_bonus_discounted = np.zeros(self.num_envs)
        if reward_shaping in ["RIDE"]:
            self.dicts_state = [dict() for _ in range(self.num_envs)]
        # Setup arrays to hold current observation and timestep
        # for each environment
        self.obss = []
        self.ts = np.array([0 for _ in range(self.num_envs)])

        # Spin up subprocesses
        self.locals = []
        self.processes = []
        self.start_processes()

    def __len__(self):
        return self.num_envs

    def __str__(self):
        self.locals[0].send(("__str__", None, None))
        return f"<ParallelShapedEnv<{self.locals[0].recv()}>>"

    def __del__(self):
        for p in self.processes:
            p.terminate()

    def gen_obs(self):
        return self.obss

    def render(self, mode="rgb_array", highlight=False):
        """Render a single environment"""
        if "BabyPANDA" in self.spec_id:
            self.locals[0].send(("render_one", mode, None))
        else:
            self.locals[0].send(("render_one", (mode, highlight), None))
        return self.locals[0].recv()

    def start_processes(self):
        """Spin up the num_envs/envs_per_proc number of processes"""
        logger.info(f"spinning up {self.num_envs} processes")
        for i in range(0, self.num_envs, self.envs_per_proc):
            local, remote = Pipe()
            self.locals.append(local)
            if "BabyPANDA" in self.spec_id:
                p = Process(target=multi_worker_cont,
                            args=(remote, self.envs[i:i + self.envs_per_proc]))
            else:
                p = Process(target=multi_worker,
                            args=(remote, self.envs[i:i + self.envs_per_proc]))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)
        logger.info("done spinning up processes")

    def request_reset_envs(self):
        """Request all processes to reset their envs"""
        logger.info("requesting resets")
        for local in self.locals:
            local.send(("reset", None, None))
        self.obss = []
        logger.info("requested resets")
        for local in self.locals:
            self.obss += local.recv()
        logger.info("completed resets")

    def reset(self):
        """Reset all environments"""
        self.request_reset_envs()
        return [obs for obs in self.obss]

    def request_step(self, actions, stop_mask):
        """Request processes to step corresponding to (primitive) actions
           unless stop mask indicates otherwise"""
        for i in range(0, self.num_envs, self.envs_per_proc):
            self.locals[i // self.envs_per_proc].send(
                ("step", [actions[i:i + self.envs_per_proc],
                          stop_mask[i:i + self.envs_per_proc]])
            )
        results = []
        for i in range(0, self.num_envs, self.envs_per_proc):
            res = self.locals[i // self.envs_per_proc].recv()
            for j in range(len(res)):
                results.append(res[j])
                if results[-1][0] != None:
                    self.obss[i + j] = results[-1][0]
        return zip(*results)

    def reset_pi_l(self):
        """Clear pi_l's memory (in case it is a recurrent policy)"""
        self.pi_l.analyze_feedback(None, 1)
        self.pi_l.on_reset()

    def reset_pi_l_partial(self, reset_mask):
        """Clear pi_l's memory for certain environments based on reset mask"""
        self.pi_l.analyze_feedback(None,
                                   torch.tensor(reset_mask).to(self.device).int().unsqueeze(1))

    def pop_masked(self, stacks, mask, allow_zero=False):
        if allow_zero:
            stacks = [stacks[i][1:] if mask[i]
                      else stacks[i] for i in range(len(stacks))]
        else:
            stacks = [stacks[i][1:] if len(stacks[i]) > 1 and mask[i]
                      else stacks[i] for i in range(len(stacks))]
        return stacks

    def step(self, actions):
        """Complete a step and evaluate low-level policy / termination
           classifier as needed depending on reward shaping scheme.
           
           Returns:  obs: list of environment observations,
                     reward: np.array of extrinsic rewards,
                     done: np.array of booleans,
                     info: depends on self.reward_shaping. Output can be used
                           to shape the reward.
        """
        # Make sure input is numpy array
        if type(actions) != np.ndarray:
            if type(actions) == list or type(actions) == int:
                actions = np.array(actions)
            elif type(actions) == torch.Tensor:
                actions = actions.cpu().numpy()
            else:
                raise TypeError
        actions_to_take = actions.copy()

        # Oracle
        if self.reward_shaping in ["subtask_oracle_ordered"]:
            self.pi_l_obss = deepcopy(self.obss)
            self.out_of_instr = np.array([False for _ in range(self.num_envs)])
            for i in range(self.num_envs):
                # For every newly reset environment, get a new stack
                if self.ts[i] == 0:
                    old_mission = self.pi_l_obss[i]['mission']
                    self.stacks[i] = self.instr_handler.get_oracle_stack(
                        old_mission, unlock="Unlock" in self.env_name)
                # For every environment, set change the mission of the
                # observation to pi_l to what's at the top of the stack
                if len(self.stacks[i]) > 0:
                    self.pi_l_obss[i]['mission'] = self.stacks[i][0]
                else:
                    self.out_of_instr[i] = True
            # Run pi_l on these observations and determine which
            # predict termination (ignoring those where the stack's empty)
            pi_l_eval = self.pi_l.act_batch(self.pi_l_obss,
                                            stop_mask=self.out_of_instr)
            pi_l_actions = pi_l_eval['action'].cpu().numpy()
            pi_l_done = (pi_l_actions == self.done_action) * \
                        (1 - self.out_of_instr)

        # LEARN Baseline
        elif self.reward_shaping in ["learn_baseline"]:
            for i in range(self.num_envs):
                if self.ts[i] == 0:
                    self.action_freqs[i] *= 0
            task_text = [self.obss[i]["mission"] for i in range(self.num_envs)]

        # Subtask classifier, static or learned online
        elif self.reward_shaping in ["subtask_classifier_static",
                                     "subtask_classifier_online",
                                     "subtask_classifier_static_unclipped",
                                     "subtask_classifier_online_unclipped",
                                     ]:
            self.pi_l_obss = [deepcopy(self.obss[i])
                              for i in range(self.num_envs)
                              for _ in range(self.num_subtasks)]
            for i in range(self.num_envs):
                # For every newly reset environment, add to the dataset
                # if task was successful (and classifier is learned online),
                # and reset arrays
                if self.ts[i] == 0:
                    old_mission = self.tasks_instr[i]
                    if self.reward_shaping in ["subtask_classifier_online",
                                               "subtask_classifier_online_unclipped"]:
                        if self.tasks_succeeded[i] and \
                                self.pi_l_already_done_all[i].sum() > 0:
                            self.subtask_online_ds.add_demos([
                                (old_mission,
                                 [(-1, np.where(self.pi_l_already_done_all[i])[0])]
                                 )
                            ])
                    self.pi_l_already_done_relevant[i] *= False
                    self.pi_l_already_done_all[i] *= False
                    self.past_pi_l_done_discounted[i] *= 0
                    self.tasks_succeeded[i] = False
                    self.tasks_instr[i] = self.obss[i]["mission"]
                # For every (environment, subtask) pair, set the mission
                # of pi_l's observation to the subtask instruction
                for j in range(self.num_subtasks):
                    self.pi_l_obss[i * self.num_subtasks + j]["mission"] = \
                        self.instr_handler.get_instruction(j)
            pi_l_eval = self.pi_l.act_batch(self.pi_l_obss, stop_mask=None)
            pi_l_actions = pi_l_eval["action"].cpu().numpy()
            pi_l_done = pi_l_actions == self.done_action
            pi_l_done = pi_l_done.reshape((self.num_envs, self.num_subtasks))
            # Just keep the instructions that weren't already done and relevant
            pi_l_done *= np.invert(self.pi_l_already_done_relevant)
            if pi_l_done.sum() > 0:
                # Preprocess the instructions for the tasks and completed
                # subtasks
                task_idx, subtask_idx = np.where(pi_l_done)
                task_text = [self.obss[i]["mission"] for i in task_idx]
                subtask_text = self.instr_handler.missions[subtask_idx]
                task_preproc = self.subtask_cls_preproc(task_text)
                subtask_preproc = self.subtask_cls_preproc(subtask_text)
                if self.reward_shaping in ["subtask_classifier_online",
                                           "subtask_classifier_online_unclipped"]:
                    task_preproc = task_preproc.to(self.device)
                    subtask_preproc = subtask_preproc.to(self.device)
                # Run them through the subtask classifier
                predicted_subtasks = self.subtask_cls(task_preproc, subtask_preproc) \
                    .round().detach().cpu().numpy().astype(bool)
                # Record them
                self.pi_l_already_done_all |= pi_l_done
                # Overwrite pi_l_done with only the done and relevant subtasks
                # and record them
                pi_l_done &= False
                for j in range(len(task_idx)):
                    if predicted_subtasks[j]:
                        pi_l_done[task_idx[j], subtask_idx[j]] = True
                        self.pi_l_already_done_relevant[task_idx[j],
                                                        subtask_idx[j]] = True

        # QG_QA
        elif self.reward_shaping in ["QG_QA"]:
            self.obss = [deepcopy(self.obss[i])
                         for i in range(self.num_envs)]
            for i in range(self.num_envs):
                # For every newly reset environment,
                # generate question and answers update mask QA and reward
                if self.ts[i] == 0:
                    # QG generation
                    mission = self.obss[i]["mission"]
                    mission = nlp(mission)
                    mission = [preprocess_token(token) for token in mission if
                               (not token.is_punct and not token.like_num)]

                    qg_data = QG(mission, self.questionable_words)
                    # numericalized
                    self.questions[i] = [numericalize(self.vocabulary['question'], x) for x in qg_data['questions']]
                    self.questions_len_begin[i] = len(self.questions[i])
                    self.questions_answered_len_current[i] = 0
                    self.answers[i] = [numericalize(self.vocabulary['answer'], [x]) for x in qg_data['answers']]

                    self.frames[i] = []
                    self.actions[i] = []
                    self.rewards_bonus[i] = 0
                    self.rewards_bonus_discounted[i] = 0

        elif self.reward_shaping in ["RIDE"]:
            self.obss = [deepcopy(self.obss[i])
                         for i in range(self.num_envs)]
            for i in range(self.num_envs):
                # For every newly reset environment,
                # generate question and answers update mask QA and reward
                if self.ts[i] == 0:
                    self.dicts_state[i] = dict()
            number_visits = torch.ones(self.num_envs, device=self.device)

        # Compute the bonus reward using QA
        if self.reward_shaping in ["QG_QA"]:
            demo_dict = {'questions': self.questions,
                         'answers': self.answers,
                         'frames': [],
                         'length_frames': [],
                         'length_frames_max': None,
                         'actions': []}

            for i in range(self.num_envs):
                im = self.obss[i]['image']
                # reordering to obtain images under the format CxHxW
                self.frames[i].append(np.array([im[:, :, 0], im[:, :, 1], im[:, :, 2]], dtype=np.uint8))
                demo_dict['frames'].append(np.array(self.frames[i]))
                demo_dict['length_frames'].append(len(self.frames[i]))
                self.actions[i].append(actions_to_take[i] + 1)  # shift needed sice 0 is the pad for action

            demo_dict['actions'] = self.actions
            demo_dict['length_frames'] = torch.tensor(demo_dict['length_frames'])

            if self.type_QG_QA_reward in ["simple"]:
                demo_batch = generate_batch(demo_dict, self.num_envs)
                with torch.no_grad():
                    answer_pred = self.QA.forward(self.vocabulary['question'], **demo_batch)[
                            'answers'].cpu().detach()
                success_pred_batch = (torch.argmax(answer_pred, dim=1) == demo_batch['answers'])
                not_answered_question = (torch.argmax(answer_pred, dim=1) != demo_batch['answers']).cpu().detach()
                # Take unanswered questions to ask them anew
                count = 0

                for i in range(self.num_envs):
                    lenq = len(self.questions[i])
                    self.rewards_bonus[i] = success_pred_batch[
                                            count:(count + lenq)].sum().cpu().detach()
                    for j in reversed(range(lenq)):
                        if not not_answered_question[count + j]:
                            del self.questions[i][j]
                            del self.answers[i][j]
                            self.questions_answered_len_current[i] += 1
                    count += lenq

            elif self.type_QG_QA_reward in ["adjusted"]:
                demo_batch = generate_batch(demo_dict, self.num_envs)
                with torch.no_grad():
                    answer_pred = self.QA.forward(self.vocabulary['question'], **demo_batch)[
                            'answers'].cpu().detach()

                good_answers = (torch.argmax(answer_pred, dim=1) == demo_batch['answers'])
                # calculate confidence of the QA in the given answer
                argm = torch.argmax(answer_pred, dim=1).unsqueeze(dim=1).detach()
                confidence = F.softmax(answer_pred, dim=1).gather(1, argm).squeeze(dim=1).cpu().detach()
                success_pred_batch = (good_answers * confidence)
                not_answered_question = (torch.argmax(answer_pred, dim=1) != demo_batch['answers']).cpu().detach()
                # Take unanswered questions to ask them anew
                count = 0

                for i in range(self.num_envs):
                    lenq = len(self.questions[i])
                    self.rewards_bonus[i] = success_pred_batch[
                                            count:(count + lenq)].sum().cpu().detach()
                    for j in reversed(range(lenq)):
                        if not not_answered_question[count + j]:
                            del self.questions[i][j]
                            del self.answers[i][j]
                            self.questions_answered_len_current[i] += 1
                    count += lenq

        if self.reward_shaping in ["RIDE"]:
            obs_state = self.obss_preprocessor(self.obss, device=self.device)
            for i in range(self.num_envs):
                key = tuple(torch.cat([obs_state.image[i].view(-1), obs_state.instr[i].view(-1)]).tolist())
                if key in self.dicts_state[i]:
                    self.dicts_state[i][key] += 1
                else:
                    self.dicts_state[i].update({key: 1})
                number_visits[i] *= self.dicts_state[i][key]

        # Make a step in the environment
        stop_mask = np.array([False for _ in range(self.num_envs)])
        obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
        reward = np.array(reward)
        done_mask = np.array(done)

        # Add reward shaping information to info
        if self.reward_shaping in ["subtask_oracle_ordered"]:
            self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
            to_reset = done | pi_l_done
            self.reset_pi_l_partial(to_reset)
            info = (pi_l_done.astype(int),
                    torch.tensor(pi_l_actions).to(self.device))

        elif self.reward_shaping in ["learn_baseline"]:
            prev_action_freqs = torch.as_tensor(
                np.nan_to_num(np.divide(self.action_freqs, self.ts[:, None]),
                              posinf=0)).float().to(self.device)
            for i in range(self.num_envs):
                self.action_freqs[i][actions_to_take[i]] += 1
            cur_action_freqs = torch.as_tensor(
                np.divide(self.action_freqs, self.ts[:, None] + 1)).float().to(self.device)
            task_preproc = self.learn_baseline_preproc(task_text).to(self.device)
            prev_pred = F.softmax(self.learn_baseline_cls(task_preproc, \
                                                          prev_action_freqs)[1], dim=-1)
            cur_pred = F.softmax(self.learn_baseline_cls(task_preproc, \
                                                         cur_action_freqs)[1], dim=-1)
            prev_potential = prev_pred[:, 1] - prev_pred[:, 0]
            cur_potential = cur_pred[:, 1] - cur_pred[:, 0]
            info = (np.stack((prev_potential.detach().cpu().numpy(), \
                              cur_potential.detach().cpu().numpy()), axis=-1), None)

        elif self.reward_shaping in ["subtask_classifier_static",
                                     "subtask_classifier_online"]:
            # Reset all pi_l models for an environment if any subtasks
            # predict termination
            to_reset = (done | pi_l_done.sum(1) > 0).repeat(self.num_subtasks)
            self.reset_pi_l_partial(to_reset)

            pi_l_done_count_clipped = pi_l_done.sum(1).clip(0, 1)
            self.past_pi_l_done_discounted += pi_l_done_count_clipped
            info = (np.stack((pi_l_done_count_clipped, self.past_pi_l_done_discounted), axis=-1),
                    torch.tensor(pi_l_actions).to(self.device))
            self.tasks_succeeded = reward > 0
            self.past_pi_l_done_discounted *= 1. / self.subtask_discount

        elif self.reward_shaping in ["QG_QA"]:
            self.rewards_bonus_discounted += self.rewards_bonus
            info = {"reward_QG_QA": np.stack((self.rewards_bonus, self.rewards_bonus_discounted), axis=-1)}
            self.rewards_bonus_discounted *= 1. / self.subtask_discount

            info["success_rate_QA"] = (self.questions_answered_len_current / self.questions_len_begin) * done_mask

        elif self.reward_shaping in ["subtask_classifier_static_unclipped",
                                     "subtask_classifier_online_unclipped"]:
            # Reset all pi_l models for an environment if any subtasks
            # predict termination
            to_reset = (done | pi_l_done.sum(1) > 0).repeat(self.num_subtasks)
            self.reset_pi_l_partial(to_reset)

            pi_l_done_count = pi_l_done.sum(1)
            self.past_pi_l_done_discounted += pi_l_done_count
            info = (np.stack((pi_l_done_count, self.past_pi_l_done_discounted), axis=-1),
                    torch.tensor(pi_l_actions).to(self.device))
            self.tasks_succeeded = reward > 0
            self.past_pi_l_done_discounted *= 1. / self.subtask_discount

        elif self.reward_shaping in ["RIDE"]:
            obs_next_state = self.obss_preprocessor(self.obss, device=self.device)
            with torch.no_grad():
                phi2_pred, actions_pred, phi1, phi2 = self.stateactionpredictor(obs_state, obs_next_state,
                                                                                torch.tensor(actions_to_take,
                                                                                             device=self.device,
                                                                                             dtype=torch.long))
            info = torch.clamp(torch.square(torch.norm(phi2 - phi1, dim=1)) / torch.sqrt(number_visits), max=1)

        self.ts += 1
        self.ts[done_mask] *= 0

        return [obs for obs in self.obss], reward, done_mask, info
