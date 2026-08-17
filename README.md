# MiniGrid RL Experiments

This project is basically my attempt to learn reinforcement learning by building a few small experiments in MiniGrid. I wanted to compare Q-learning and SARSA on a simple navigation problem and see how the agent behaves when the environment is fixed versus randomized.

## What I was trying to do

The main idea was to train an agent to move through a simple crossing-like maze and reach the goal. I was curious about a few things:

- does the agent learn better in a fixed environment?
- does randomizing the maze or positions make learning harder or more realistic?
- how do Q-learning and SARSA compare?
- what hyperparameters actually matter?

This ended up being a small RL experiment project rather than a polished app.

## Environment

The environment is based on MiniGrid, using something like:

- MiniGrid-SimpleCrossingS9N1-v0

The agent has to move around a simple grid and reach a goal while avoiding obstacles. I reduced the state to a simple representation using:

- x position
- y position
- direction

So the state becomes a single encoded index that I use in a Q-table.

## Algorithms used

### Q-learning

This was the main one I used to get the agent learning from experience. It updates based on the best estimated future reward from the next state.

### SARSA

I also tried SARSA as the on-policy version of the same idea. The difference is that it uses the action actually taken next in the update, which makes it a little different behavior-wise.

---

## Files in this project

### Main training scripts

- [fixed_Q.py](fixed_Q.py): Q-learning with a fixed environment and fixed seed.
- [fixed_SARSA.py](fixed_SARSA.py): SARSA with the same fixed environment setup.
- [partial_random_Q.py](partial_random_Q.py): Q-learning with some randomization added.
- [partial_random_SARSA.py](partial_random_SARSA.py): SARSA with some randomization added.
- [total_random_Q.py](total_random_Q.py): Q-learning in a more fully randomized environment.
- [total_random_SARSA.py](total_random_SARSA.py): SARSA in a more fully randomized environment.

### Extra experiment files

- [hyperparameter_tuning.py](hyperparameter_tuning.py): tries different values for learning rate, gamma, and epsilon decay.
- [simple_path_visualization.py](simple_path_visualization.py): visualizes the path the trained agent takes through the environment.

### Tests / debugging files

- [test_fixed.py](test_fixed.py): checks whether the environment stays deterministic when seeded.
- [test_randomization.py](test_randomization.py): checks whether randomization is actually happening as expected.

---

## How the scripts work

Most of the training files follow the same pattern:

1. create the MiniGrid environment
2. build the Q-table
3. reset the environment each episode
4. choose an action using epsilon-greedy exploration
5. take the action and observe the reward
6. update the Q-table
7. decay epsilon over time
8. record the reward and success rate
9. plot the results

It’s a pretty standard tabular RL setup, but I wanted to explore how the environment setup changes the outcome.

---

## Dependencies

I used a few Python packages for this:

- Python
- NumPy
- Matplotlib
- MiniGrid
- Gymnasium or Gym, depending on the script

Some files use the older gym + gym_minigrid setup, while others use the newer gymnasium + minigrid setup.

---

## Setup

If you want to run it yourself, you might need:

```bash
pip install numpy matplotlib gymnasium minigrid
```

Or, if you’re using the older setup:

```bash
pip install gym gym-minigrid
```

---

## How to run it

Most scripts can just be run directly:

```bash
python fixed_Q.py
```

```bash
python fixed_SARSA.py
```

```bash
python hyperparameter_tuning.py
```

```bash
python simple_path_visualization.py
```

These usually print results and generate plots.

---

## What I was looking at in the results

The output from each script is mostly:

- average reward per episode
- success rate over time
- how quickly the agent improves
- comparison between methods and conditions

I was mainly trying to see whether learning becomes more stable in a fixed environment, and whether adding randomness makes the agent more robust or just harder to train.

---

## Final thoughts

This project is not super formal or production-level, but it was a good way for me to learn RL hands-on. I got to work with:

- Q-learning
- SARSA
- state encoding
- epsilon-greedy policies
- hyperparameter tuning
- randomized environment design
- basic result plotting

It ended up being a nice personal project to explore how reinforcement learning behaves in a simple grid world.

If I had to describe it in one sentence: it’s a small self-directed RL project comparing learning behavior in fixed and randomized MiniGrid environments.
