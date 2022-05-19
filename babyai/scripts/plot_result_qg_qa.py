import numpy as np
from matplotlib import pyplot as plt


def plot_d_R():
    paths = ['storage/QG_QA_evaluation/PNR_cumulative_average_Reward_compare_bot_agent_rand_0_to_1_valid.npy',
             'storage/QG_QA_evaluation/PNR_Simple_Reward_compare_bot_agent_rand_0_to_1_valid.npy',
             'storage/QG_QA_evaluation/PNR_Adjust_Reward_compare_bot_agent_rand_0_to_1_valid.npy']
    labels = ["cumulative_average",
              "simple",
              "adjust"]
    proba = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    for p, l in zip(paths, labels):
        with open(p, 'rb') as f:
            average_reward = np.load(f)
            d_average_reward = np.diff(average_reward) / 0.1
            print(d_average_reward)
            plt.plot(proba, d_average_reward, label=l)

    plt.legend(loc='best')
    plt.title("comparison for the derivative of the reward for different reward strategies")
    plt.show()


def plot_average_first_answer_step_PNR():
    proba = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])

    step = np.array([[9.237, 10.386, 12.188, 14.388, 17.046, 20.807, 28.308, 40.383, 50.914, 55.369, 50.589],
                     [10.247, 11.676, 13.499, 16.627, 20.501, 26.115, 37.507, 52.446, 65.166, 70.999, 71.130],
                     [11.008, 12.806, 14.630, 17.596, 21.106, 25.576, 34.420, 45.905, 57.415, 65.084, 67.651],
                     [12.243, 14.224, 16.511, 20.242, 24.750, 31.323, 44.408, 61.954, 76.083, 81.906, 82.213]])
    labels = ["Q1",
              "Q2",
              "Q3",
              "Q4"]
    for s, l in zip(step, labels):
        plt.plot(proba, s, label=l)

    plt.legend(loc='best')
    plt.title("comparison in the average number of step before the first good answer ")
    plt.show()


plot_d_R()
# plot_average_first_answer_step_PNR()
