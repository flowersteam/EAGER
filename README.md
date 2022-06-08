# EAGER: Asking and Answering Questions for Automatic Reward Shaping in Language-guided RL



We propose an automated reward shaping method for guiding exploration in instruction following settings where the agent extracts auxiliary objectives from the general goal by asking itself intermediate questions, building on a question generation (QG) and question answering (QA) system.
## Installation
Clone this repository
```
git clone https://anonymous.4open.science/r/EAGER-FC2E/README.md
cd EAGER
```

Create a new environment and activate it

```
conda env create -n eager
conda activate eager
```

Install BabyAI and MiniGrid.

```
cd babyai
pip install -e .
cd ../gym-minigrid
pip install -e .
```

## Pretraining the QA module

# Generating demonstrations
We use a mix of the  PutNextTo, PickUp, Open, and Sequence tasks for generating training trajectories. These trajectories are associated with questions among which some are not answerable. 

```
scripts/sample_runs/gen_demos/multienv_QG.sh
```

# Train the QA

```
scripts/sample_runs/train_QA/multienv_QA.sh
```

## RL training with EAGER

For the PutNextTo setting in 1 room with QA trained to answer no_answer if necessary.

```
scripts/sample_runs/rl/pnl_QG_QA_no_answer.sh
```

## Acknowledgements

- [babyai](https://github.com/mila-iqia/babyai/tree/iclr19)
- [gym-minigrid](https://github.com/maximecb/gym-minigrid)
- [ella](https://github.com/Stanford-ILIAD/ELLA)
- [ride](https://github.com/facebookresearch/impact-driven-exploration)
