import torch
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

import babyai.utils as utils

from babyai.QA_simple import Model as Model_simple
from attrdict import AttrDict
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import blosc


def generate_batch(indices, demo, batch_size):
    offset = 0
    for batch_index in range(len(indices) // batch_size):
        batch_demo = {}
        for k in demo:
            if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                if k == 'questions':
                    batch_demo[k] = []
                    for i in indices[offset: offset + batch_size]:
                        len_q = demo[k][i].shape[0]
                        for j in range(len_q):
                            batch_demo[k].append(torch.unsqueeze(demo[k][i][j].type(torch.IntTensor), 0))
                elif k == 'answers':
                    batch_demo[k] = []
                    for i in indices[offset: offset + batch_size]:
                        len_q = len(demo[k][i])
                        for j in range(len_q):
                            batch_demo[k].append(demo[k][i][j])

                else:
                    batch_demo[k] = []
                    for i in indices[offset: offset + batch_size]:
                        len_q = demo['questions'][i].shape[0]
                        for j in range(len_q):
                            if k == 'actions':
                                batch_demo[k].append(torch.unsqueeze(demo[k][i].type(torch.IntTensor), 0))
                            elif k == 'frames':
                                batch_demo[k].append(torch.unsqueeze(demo[k][i].type(torch.FloatTensor), 0))
                            else:
                                batch_demo[k].append(torch.unsqueeze(torch.unsqueeze(demo[k][i], 0), 0))

                if k == 'questions' or k == 'frames' or k == 'actions' or k == 'length_frames':
                    batch_demo[k] = torch.cat(batch_demo[k], 0)
                elif k == 'answers':
                    batch_demo[k] = torch.tensor(batch_demo[k])
            else:
                batch_demo[k] = demo[k]

        assert batch_demo['questions'].shape[0] == batch_demo['answers'].shape[0]
        assert batch_demo['questions'].shape[0] == batch_demo['frames'].shape[0]
        assert batch_demo['questions'].shape[0] == batch_demo['actions'].shape[0]

        pkl.dump(batch_demo, open("storage/batch_valid/study/batch_{}.pkl".format(batch_index), "wb"))

        offset += batch_size


def learning_curves(name_env, model_number):
    print("======== env:{} model:{}=======".format(name_env, model_number))
    log = pkl.load(open('storage/models/' + name_env + '/' + 'model_{}'.format(model_number) + '/log.pkl', "rb"))

    train_error = np.array(log["loss_cross_entropy_train"])
    success_rate_train = np.array(log["success_pred_train"])
    valid_error = np.array(log["loss_cross_entropy_valid"])
    success_rate_valid = np.array(log["success_pred_valid"])

    print('At epoch {} the CE error for train reach the minimum value of {}'.format(np.argmin(train_error),
                                                                                    min(train_error)))
    print(train_error)
    print(" ")
    print('At epoch {} the CE error for valid reach the minimum value of {}'.format(np.argmin(valid_error),
                                                                                    min(valid_error)))
    print(valid_error)
    print(" ")
    print('At epoch {} the success rate for train reach the maximum value of {}'.format(np.argmax(success_rate_train),
                                                                                        max(success_rate_train)))
    print(success_rate_train)
    print(" ")
    print('At epoch {} the success rate for valid reach the maximum value of {}'.format(np.argmax(success_rate_valid),
                                                                                        max(success_rate_valid)))
    print(success_rate_valid)

    """plt.plot(np.arange(len(train_error)), train_error)
    plt.title("Train error")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(valid_error)), valid_error)
    plt.title("Valid error")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(success_rate_train)), success_rate_train)
    plt.title("Success rate train set")
    plt.grid(axis='both')
    plt.show()
    plt.plot(np.arange(len(success_rate_valid)), success_rate_valid)
    plt.title("Success rate valid set")
    plt.grid(axis='both')
    plt.show()
"""

def generate_batch(indices, batch_size, demo, train=True):

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

def studying_QA(test_env, train_env, model, epoch):

    demos_path_valid = utils.get_demos_QG_path('{}_agent_done'.format(test_env),
                                               None, None, valid=True)
    demos_path_valid = demos_path_valid.replace(test_env, 'test_multienv2/'+test_env)
    print(demos_path_valid)
    demos = utils.load_demos(demos_path_valid)

    demo_voc = utils.get_demos_QG_voc_path('{}_agent_done'.format(train_env), None, None, valid=False)
    demo_voc = demo_voc.replace('QG', 'QG_no_answer')
    print(demo_voc)
    vocab = utils.load_voc(demo_voc)
    # values for the model
    emb_size = len(vocab['question'])
    print(vocab['question'].to_dict()['index2word'])
    vocab_list = vocab['answer'].to_dict()['index2word']
    vocab_list[0] = "no_answer"
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
    attr['vocab_path'] = 'storage/demos/{}_agent_done_QG_no_answer_vocab.pkl'.format(train_env)
    print(attr['vocab_path'])
    et_qa = Model_simple(attr, emb_size, numb_action, pad=0)
    print('storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(train_env, model, epoch))
    et_qa.load_state_dict(torch.load('storage/models/{}_no_answer/model_{}/et_qa_{}.pt'.format(train_env,
                                                                                               model,
                                                                                               epoch)))
    et_qa.cuda()
    et_qa.eval()
    generated = False

    with torch.no_grad():
        indices = list(range(len(demos['questions'])))
        np.random.shuffle(indices)

        batch_size = min(5, len(demos['questions']))

        if not generated:
            generated = True
            np.random.shuffle(indices)
            generate_batch(indices=indices, batch_size=batch_size, demo=demos, train=False)

        desc = 'valid demos'
        with tqdm(total=len(indices) // batch_size, desc=desc) as t:
            success_pred_batch = 0
            answer_loss_batch = 0
            count_epoch_pred = np.zeros(len(vocab['answer']))
            count_epoch_true = np.zeros(len(vocab['answer']))
            for batch_index in range(len(indices) // batch_size):
                demo = pkl.load(open("storage/batch_valid/batch_{}.pkl".format(batch_index), "rb"))
                batch_demo = dict()
                for k, v in demo.items():
                    if k != 'length_frames_max' and k != 'env_ids' and k != 'missions':
                        # print(k)
                        batch_demo[k] = v.cuda()
                    else:
                        batch_demo[k] = v
                answer_pred = et_qa.forward(vocab['question'], **batch_demo)
                answer_loss = F.cross_entropy(answer_pred['answers'], batch_demo['answers'], reduction='mean')
                answer_loss_batch += answer_loss.cpu().detach().numpy()
                # print(batch_demo['questions'][:2])
                unique, return_counts = torch.unique(torch.argmax(answer_pred['answers'], dim=1), return_counts=True)
                l_u = len(unique)
                for u in range(l_u):
                    unique_u = unique[u].cpu().detach().numpy()
                    count_epoch_pred[unique_u] += return_counts[u].cpu().detach().numpy()
                unique, return_counts = torch.unique(batch_demo['answers'], return_counts=True)
                l_u = len(unique)
                for u in range(l_u):
                    unique_u = unique[u].cpu().detach().numpy()
                    count_epoch_true[unique_u] += return_counts[u].cpu().detach().numpy()

                success_pred_batch += (torch.argmax(answer_pred['answers'], dim=1)
                                       == batch_demo['answers']).sum().cpu().detach().numpy() / \
                                      batch_demo['answers'].shape[0]
                # print(success_pred_batch)
                t.update()
            pred_percent_class = count_epoch_pred / np.sum(count_epoch_pred)
            true_percent_class = count_epoch_true / np.sum(count_epoch_true)
            print('Prediction percentage for each class {}'.format(pred_percent_class))
            print('True percentage for each class {}'.format(true_percent_class))
            sr = success_pred_batch / (len(indices) // batch_size)
            print(sr)
            print(answer_loss_batch / (len(indices) // batch_size))
            plt.figure(figsize=(10,5))

            # Width of a bar
            width = 0.3
            ind = np.arange(len(vocab_list))
            # Plotting
            plt.bar(ind, pred_percent_class, width, label='Prediction percentages')
            plt.bar(ind + width, true_percent_class, width, label='True percentages')

            plt.xlabel('Words')
            plt.ylabel('Percentage')
            name_env = test_env[7:-3]
            plt.title('Percentage predicted words VS true percentage on {}, QA train on multienv2, SR: {}'.format(name_env,round(sr, 4)))

            # xticks()
            # First argument - A list of positions at which ticks should be placed
            # Second argument -  A list of labels to place at the given locations
            plt.xticks(ind + width / 2, vocab_list, rotation=20)

            # Finding the best position for legends and putting it
            plt.legend(loc='best')
            plt.show()


# learning_curves('BabyAI-PutNextLocal-v0', 0)
# learning_curves('BabyAI-PutNextLocal-v0_no_answer', 1)
# learning_curves('BabyAI-PutNextLocal-v0_no_answer', 2)
# learning_curves('multienv', 0)
# learning_curves('multienv', 1)
# learning_curves('multienv_no_answer', 0)
# learning_curves('multienv2_no_answer', 0)
learning_curves('BabyAI-PutNextLocal-v0_no_answer_biased', 0)
learning_curves('BabyAI-PutNextLocal-v0_no_answer_l_class', 0)


"""envs = ['BabyAI-SynthThenSynthMedium-v0',
        'BabyAI-Unlock-v0',
        'BabyAI-Pickup-v0', 'BabyAI-PutNextLocal-v0',
        'BabyAI-PutNextMedium-v0', 'BabyAI-PutNextLarge-v0']

for e in envs:
   studying_QA(e, 'multienv2', 0, 7)"""
# studying_QA('BabyAI-SynthThenSynthMedium-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-Unlock-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-Pickup-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-PutNextLocal-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-PutNextMedium-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-PutNextLarge-v0', 'multienv2', 0, 7)
# studying_QA('BabyAI-OpenAndPickupMedium-v0', 'multienv3', 0, 8)
"""studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd04', 'multienv3', 0, 8)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd04', 'multienv3', 0, 10)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd06', 'multienv3', 0, 8)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd06', 'multienv3', 0, 10)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd07', 'multienv3', 0, 8)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd07', 'multienv3', 0, 10)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd08', 'multienv3', 0, 8)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd08', 'multienv3', 0, 10)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd09', 'multienv3', 0, 8)
studying_QA('BabyAI-SynthThenSynthMedium-v0_rnd09', 'multienv3', 0, 10)"""
# studying_QA('BabyAI-UnlockMedium-v0', 'multienv2', 0, 7)