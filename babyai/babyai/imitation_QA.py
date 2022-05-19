import copy
import gym
import time
import datetime
import gc
import numpy as np
import sys
import itertools
import torch
import pickle as pkl
import blosc
import multiprocessing
import os
import logging
import babyai.utils as utils

import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from PIL import Image
from gym import spaces

from babyai.evaluate import batch_evaluate
from babyai.QA import Model
from babyai.QA_simple import Model as Model_simple

logger = logging.getLogger(__name__)

softmax = nn.Softmax(dim=1)


class ImitationLearning(object):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        utils.seed(self.args.seed)

        demos_path = utils.get_demos_QG_path(args.demos, args.env, args.demos_origin, valid=False)
        demos_path_valid = utils.get_demos_QG_path(args.demos, args.env, args.demos_origin, valid=True)
        demos_voc = utils.get_demos_QG_voc_path(args.demos, args.env, args.demos_origin, valid=False)

        if self.args.no_answer_question:
            if self.args.biased and self.args.env == "BabyAI-PutNextLocal-v0":
                demos_path = str(demos_path).replace("QG", "QG_no_answer_biased")
                demos_path_valid = str(demos_path_valid).replace("QG", "QG_no_answer_biased")
                demos_voc = str(demos_voc).replace("QG", "QG_no_answer_biased")
                print(demos_path)
                print(demos_path_valid)
                print(demos_voc)
                logger.info('loading train demos')
                self.train_demos = utils.load_demos(demos_path)
                logger.info('loaded train demos')

                logger.info('loading valid demos')
                self.valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded valid demos')

                logger.info('loading voc train demos')
                self.demos_voc = utils.load_voc(demos_voc)
                logger.info('loaded voc train demos')
                """
                demos_path_l_class = str(demos_path).replace("QG", "QG_no_answer_biased_l_class")
                demos_path_valid_l_class = str(demos_path_valid).replace("QG", "QG_no_answer_biased_l_class")
                demos_voc_l_class = str(demos_voc).replace("QG", "QG_no_answer_biased_l_class")
                print(demos_path_l_class)
                print(demos_path_valid_l_class)
                print(demos_voc_l_class)
                logger.info('loading train demos language classifier')
                self.train_demos_l_class = utils.load_demos(demos_path_l_class)
                logger.info('loaded train demos language classifier')
        
                logger.info('loading valid demos language classifier')
                self.valid_demos_l_class = utils.load_demos(demos_path_valid_l_class)
                logger.info('loaded valid demos language classifier')
        
                logger.info('loading voc train demos language classifier')
                self.demos_voc_l_class = utils.load_voc(demos_voc_l_class)
                logger.info('loaded voc train demos language classifier')"""

            else:
                demos_path = str(demos_path).replace("QG", "QG_no_answer")
                demos_path_valid = str(demos_path_valid).replace("QG", "QG_no_answer")
                demos_voc = str(demos_voc).replace("QG", "QG_no_answer")
                print(demos_path)
                print(demos_path_valid)
                print(demos_voc)

                logger.info('loading train demos')
                self.train_demos = utils.load_demos(demos_path)
                logger.info('loaded train demos')

                logger.info('loading valid demos')
                self.valid_demos = utils.load_demos(demos_path_valid)
                logger.info('loaded valid demos')

                logger.info('loading voc train demos')
                self.demos_voc = utils.load_voc(demos_voc)
                logger.info('loaded voc train demos')

        else:
            logger.info('loading train demos')
            self.train_demos = utils.load_demos(demos_path)
            logger.info('loaded train demos')

            logger.info('loading valid demos')
            self.valid_demos = utils.load_demos(demos_path_valid)
            logger.info('loaded valid demos')

            logger.info('loading voc train demos')
            self.demos_voc = utils.load_voc(demos_voc)
            logger.info('loaded voc train demos')


        """
        if args.episodes:
            if args.episodes > len(self.train_demos['questions']):
                raise ValueError("there are only {} train demos".format(len(self.train_demos['questions'])))
        
            self.train_demos = {self.train_demos[k]: (self.train_demos[k][:args.episodes] if k != 'length_frames_max' else self.train_demos[k]) for k in self.train_demos}

        if args.val_episodes > len(self.valid_demos['questions']):
            logger.info('Using all the available {} demos to evaluate valid. accuracy'.format(len(self.valid_demos['questions'])))
        self.valid_demos = {self.valid_demos[k]: (self.valid_demos[k][:self.args.val_episodes] if k != 'length_frames_max' else self.valid_demos[k]) for k in self.valid_demos}
        """

        # Define episodic transformer for QA
        if self.args.QA:
            emb_size = len(self.demos_voc['question'])
            # action: pad_value=0, left=1, right=2, forward=3, pickup=4, drop=5, toggle=6 done=7
            numb_action = 8
            if self.args.simple_embedder:
                self.et_qa = Model_simple(attr, emb_size, numb_action, pad=0)
            else:
                self.et_qa = Model(attr, emb_size, numb_action, pad=0)
        else:
            ValueError("no args.QA")

        if torch.cuda.is_available():
            self.et_qa.cuda()

        self.optimizer = torch.optim.Adam(self.et_qa.parameters(),
                                          self.args.lr,
                                          eps=self.args.optim_eps)
        self.scheduler_1 = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                           step_size=4,
                                                           gamma=0.1)

        """self.scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                      mode='min',
                                                                      factor=0.1,
                                                                      patience=2)"""
        """self.scheduler_3 = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                           step_size=5,
                                                           gamma=0.1)"""
        """self.scheduler_seq = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                                   schedulers=[self.scheduler_1, self.scheduler_3],
                                                                   milestones=[10])"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_batch(self, indices, batch_size, train=True):
        if train:
            demo = self.train_demos
        else:
            demo = self.valid_demos

        offset = 0
        with tqdm(total=len(indices) // batch_size, desc="creation batch") as t:
            for batch_index in range(len(indices) // batch_size):
                batch_demo = {}
                for k in demo:
                    if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                        if k == 'questions':
                            batch_demo[k] = []
                            # print('k == questions')
                            # print(demo[k][0])
                            # print(demo[k][0].shape[0])
                            for i in indices[offset: offset + batch_size]:
                                len_q = len(demo[k][i])
                                for j in range(len_q):
                                    batch_demo[k].append(demo[k][i][j])
                            # print(batch_demo[k][:4])
                        elif k == 'answers':
                            batch_demo[k] = []
                            # print('k == answers')
                            # print(demo[k][0])
                            for i in indices[offset: offset + batch_size]:
                                len_q = len(demo[k][i])
                                for j in range(len_q):
                                    batch_demo[k].append(demo[k][i][j])
                            # print(batch_demo[k][:4])

                        else:
                            batch_demo[k] = []
                            for i in indices[offset: offset + batch_size]:
                                len_q = len(demo['questions'][i])
                                for j in range(len_q):
                                    if k == 'actions':
                                        batch_demo[k].append(demo[k][i])
                                    elif k == 'frames':
                                        frames = blosc.unpack_array(demo[k][i])
                                        frames_tensor = torch.from_numpy(frames)
                                        # batch_demo[k].append(torch.unsqueeze(frames_tensor.type(torch.FloatTensor), 0))
                                        batch_demo[k].append(frames_tensor)
                                    else:
                                        batch_demo[k].append(torch.unsqueeze(torch.unsqueeze(demo[k][i], 0), 0))

                        if k == 'length_frames':
                            batch_demo[k] = torch.cat(batch_demo[k], 0)
                            batch_demo['length_frames_max'] = max(batch_demo[k]).cuda()
                        elif k == 'answers':
                            batch_demo[k] = torch.tensor(batch_demo[k])
                        # print(k)
                    elif k != 'length_frames_max':
                        batch_demo[k] = demo[k]

                # pad  and tensorize questions
                batch_demo['questions'] = pad_sequence(
                    [torch.tensor(x, dtype=torch.float32) for x in batch_demo['questions']],
                    batch_first=True,
                    padding_value=0).type(torch.IntTensor)

                # pad and tensorize actions
                batch_demo['actions'] = pad_sequence(
                    [torch.tensor(x, dtype=torch.float32) for x in batch_demo['actions']],
                    batch_first=True,
                    padding_value=0).type(torch.IntTensor)

                batch_demo['frames'] = pad_sequence(batch_demo['frames'], batch_first=True,
                                                        padding_value=0).detach()

                assert batch_demo['questions'].shape[0] == batch_demo['answers'].shape[0]
                assert batch_demo['questions'].shape[0] == batch_demo['actions'].shape[0]
                assert batch_demo['questions'].shape[0] == batch_demo['frames'].shape[0]

                if train:
                    pkl.dump(batch_demo, open("storage/batch_train/batch_{}.pkl".format(batch_index), "wb"))
                else:
                    pkl.dump(batch_demo, open("storage/batch_valid/batch_{}.pkl".format(batch_index), "wb"))

                offset += batch_size
                t.update()

    def train(self):

        # Log dictionary
        log = {"loss_cross_entropy_train": [], "success_pred_train": [], "loss_cross_entropy_valid": [],
               "success_pred_valid": [], "confidence": []}
        generated = False
        print(' ')
        print('Batch generated {}, simple image embedder {}'.format(generated, self.args.simple_embedder))
        print(' ')
        unique_t = []
        count_t = []

        for e in range(self.args.epochs):
            print('lr {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))
            # Train
            batch_size = min(self.args.batch_size, len(self.train_demos['questions']))
            if self.args.epoch_length == 0:
                indices = list(range(len(self.train_demos['questions'])))
            else:
                indices = np.random.choice(len(self.train_demos['questions']), self.args.epoch_length)
            self.et_qa.train()

            if not generated:
                np.random.shuffle(indices)
                self.generate_batch(indices, batch_size, train=True)

            with tqdm(total=len(indices) // batch_size, desc="train") as t:
                answer_loss_batch = 0
                success_pred_batch = 0
                for batch_index in range(len(indices) // batch_size):
                    # batch_index_overfit = 0
                    demo = pkl.load(open("storage/batch_train/batch_{}.pkl".format(batch_index), "rb"))
                    batch_demo = {}
                    for k, v in demo.items():
                        if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                            batch_demo[k] = v.cuda()
                        else:
                            batch_demo[k] = v

                    answer_pred = self.et_qa.forward(self.demos_voc['question'], **batch_demo)
                    # count the number of time a class is answered on a sub_batch
                    if batch_index == 0 or batch_index == (len(indices) // batch_size) - 1:
                        unique, return_counts = torch.unique(torch.argmax(answer_pred['answers'], dim=1),
                                                             return_counts=True)
                        unique_t.append(unique.cpu().detach().numpy())
                        d = {k: v for k, v in
                             zip(self.demos_voc['answer'].index2word(list(unique.cpu().detach().numpy())),
                                 return_counts.cpu().detach().numpy())}
                        print(d)
                        count_t.append(return_counts.cpu().detach().numpy())

                    success_pred_batch += (torch.argmax(answer_pred['answers'], dim=1)
                                           == batch_demo['answers']).sum().cpu().detach().numpy() / \
                                          batch_demo['answers'].shape[0]

                    answer_loss = F.cross_entropy(answer_pred['answers'], batch_demo['answers'], reduction='mean')
                    del answer_pred
                    gc.collect()
                    torch.cuda.empty_cache()
                    answer_loss_batch += answer_loss.cpu().detach().numpy()

                    self.optimizer.zero_grad()
                    answer_loss.backward()
                    self.optimizer.step()

                    t.update()
                self.scheduler_1.step()
                # print('lr {}'.format(self.scheduler_1.get_last_lr()))
                # self.scheduler_seq.step()

                log["loss_cross_entropy_train"].append(answer_loss_batch / (len(indices) // batch_size))
                log["success_pred_train"].append(success_pred_batch / (len(indices) // batch_size))

            # Valid
            with torch.no_grad():
                if self.args.epoch_length == 0:
                    indices = list(range(len(self.valid_demos['questions'])))
                else:
                    indices = np.random.choice(len(self.valid_demos['questions']), self.args.epoch_length)

                batch_size = min(self.args.batch_size, len(self.valid_demos['questions']))

                self.et_qa.eval()

                if not generated:
                    np.random.shuffle(indices)
                    self.generate_batch(indices, batch_size, train=False)
                    generated = True

                with tqdm(total=len(indices) // batch_size, desc="valid") as t:
                    answer_loss_batch = 0
                    success_pred_batch = 0
                    '''table_confidence = np.zeros(4)'''
                    for batch_index in range(len(indices) // batch_size):
                        demo = pkl.load(open("storage/batch_valid/batch_{}.pkl".format(batch_index), "rb"))
                        batch_demo = {}
                        for k, v in demo.items():
                            if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                                batch_demo[k] = v.cuda()
                            else:
                                batch_demo[k] = v
                        with torch.no_grad():
                            answer_pred = self.et_qa.forward(self.demos_voc['question'], **batch_demo)
                            answer_loss = F.cross_entropy(answer_pred['answers'], batch_demo['answers'], reduction='mean')

                        success_pred_batch += (torch.argmax(answer_pred['answers'], dim=1)
                                               == batch_demo['answers']).sum().cpu().detach().numpy() / \
                                              batch_demo['answers'].shape[0]

                        answer_loss_batch += answer_loss.cpu().detach().numpy()
                        """table_confidence += \
                            torch.max(softmax(answer_pred['answers']), dim=1).values.view(-1, 4).sum(dim=0).cpu().detach().numpy()/batch_size"""
                        t.update()

                    # self.scheduler_2.step(answer_loss_batch / (len(indices) // batch_size))
                    log["loss_cross_entropy_valid"].append(answer_loss_batch / (len(indices) // batch_size))
                    log["success_pred_valid"].append(success_pred_batch / (len(indices) // batch_size))
                    '''log["confidence"].append(table_confidence/ (len(indices) // batch_size))'''

            logger.info(
                'Epoch {} train CE {} SR {} valid CE {} and the SR is {}'.format(e,
                                                                                 log["loss_cross_entropy_train"][-1],
                                                                                 log["success_pred_train"][-1],
                                                                                 log["loss_cross_entropy_valid"][-1],
                                                                                 log["success_pred_valid"][-1]))
            """logger.info('Confidence {}'.format(log["confidence"][-1]))"""
            if self.args.no_answer_question:
                if self.args.biased and self.args.env == "BabyAI-PutNextLocal-v0":
                    pkl.dump(log, open(
                        'storage/models/{}_no_answer_biased/model_{}/log.pkl'.format(self.args.env, self.args.model_number), "wb"))
                    pkl.dump(np.array(unique_t, dtype=object), open(
                        'storage/models/{}_no_answer_biased/model_{}/unique.pkl'.format(self.args.env, self.args.model_number),
                        "wb"))
                    pkl.dump(np.array(count_t, dtype=object), open(
                        'storage/models/{}_no_answer_biased/model_{}/count.pkl'.format(self.args.env, self.args.model_number),
                        "wb"))
                    torch.save(self.et_qa.state_dict(),
                               'storage/models/{}_no_answer_biased/model_{}/et_qa_{}.pt'.format(self.args.env,
                                                                                         self.args.model_number, e))
                else:
                    pkl.dump(log, open(
                        'storage/models/{}_no_answer/model_{}/log.pkl'.format(self.args.env, self.args.model_number), "wb"))
                    pkl.dump(np.array(unique_t, dtype=object), open(
                        'storage/models/{}_no_answer/model_{}/unique.pkl'.format(self.args.env, self.args.model_number),
                        "wb"))
                    pkl.dump(np.array(count_t, dtype=object), open(
                        'storage/models/{}_no_answer/model_{}/count.pkl'.format(self.args.env, self.args.model_number),
                        "wb"))
                    torch.save(self.et_qa.state_dict(),
                               'storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(self.args.env,
                                                                                         self.args.model_number, e))
            else:
                pkl.dump(log,
                         open('storage/models/{}/model_{}/log.pkl'.format(self.args.env, self.args.model_number), "wb"))
                pkl.dump(np.array(unique_t, dtype=object),
                         open('storage/models/{}/model_{}/unique.pkl'.format(self.args.env, self.args.model_number),
                              "wb"))
                pkl.dump(np.array(count_t, dtype=object),
                         open('storage/models/{}/model_{}/count.pkl'.format(self.args.env, self.args.model_number),
                              "wb"))
                torch.save(self.et_qa.state_dict(),
                           'storage/models/{}/model_{}/et_qa_{}.pt'.format(self.args.env, self.args.model_number, e))

        return log
