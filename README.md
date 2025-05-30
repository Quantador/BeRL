# Preference-Based RL Experiments

This repository contains reinforcement learning experiments using preference data on environments like Acrobot and CartPole. The code supports both RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) approaches.

## Supported Environments

- `Acrobot-v1`
- `CartPole-v0`
- `MountainCarContinuous-v0` *(retained for reference but not used)*

## Project Structure

```
.
├── with-Acrobot-v1
│   ├── policies/
│   ├── pref_data/
│   └── scripts/
│
├── with-CartPole-v0
│   ├── policies/
│   ├── pref_data/
│   └── scripts/
│
├── with-MountainCarContinuous-v0
│   ├── policies/
│   ├── pref_data/
│   └── scripts/
│
├── .gitignore
└── README.md
```




## Workflow 
In each environment folder, you typically follow this order: 
 
1. `generate_pairs.ipynb`   
   Generates the preference pairs of trajectories that will be used to train using RLHF and DPO.   
   This is where you define the size of the dataset with the K parameter. 
 
2. `trainingRLHF.ipynb`   
   Trains a reward model and use it to optimize our checkpoint policy using REINFORCE based on the generated trajectories preferences. 
 
3. `training_DPO.ipynb`   
   Optimize a our checkpoint policy using Direct Preference Optimization based on the generated trajectories preferences.



