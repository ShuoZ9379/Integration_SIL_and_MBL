# Model-based Lookahead Policy for Model-free RL
This project aims to combine the MPC-style planning with model-free RL to enhance the performacne/data-efficiency of model-free RL. The preliminary experimental results show that this approach does enhance the performance of TRPO agents in the selected continuous control tasks (e.g. Ant-v2, HalfCheetah-v2, Reacher-v2, Pusher-v2, Pendulum-v0). However, we identify this method leads to negative effects in some tasks like Walker2d-v2 and Hopper-v2. The further improvements are under study.

## Installation
- Create a virtualenv or use miniconda/anaconda
- Install the dependencies
```
pip install -r requirement.txt
```
- Install mbl
```
pip install -e .
```

## Run experiment
###  TRPO
```
./run_trpo_experiment.sh {env id} {experiment id} {seed} {gpu} "{extra args}"

Example:
./run_trpo_experiment.sh Ant-v2 trial-0 0 1 "--num_samples=1500 --horizon=5 --num_elites=1 --mbl_lamb=0.5"
```

## Experimental results
### MuJoCo and Classic control
![mujoco_perf](/experiement/fig_mujoco_perf.png)

## TODO
- [x] Include preliminary experimental results (figures, log) in the repo
- [x] Add validation phase for model selection (prevent degeneration of model during training)
- [ ] Implement Probabilistic NN for forward dynamic model
- [ ] Implement Ensemble of ProNN for forward dynamic model
- [ ] Plot the performance with different H
- [ ] Plot the performance with different LAMB
- [ ] Add support for REINFORCE
- [ ] Conduct experiments on more challenging environments (e.g. Hand manipulation, real robot)
- [ ] Conduct experiments on sparse/deceptive reward settings (can our method overcome such challenging reward settings?)
