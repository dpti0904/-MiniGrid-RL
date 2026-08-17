import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import matplotlib.pyplot as plt

def encode_state_agent(env):
    """
    Encode the agent's (x, y, direction) into a unique integer
    """
    x, y = env.agent_pos
    d    = env.agent_dir

    width  = env.width
    height = env.height

    return d * (width * height) + y * width + x

def train_q_learning_fixed(num_episodes=1000,
                     alpha=0.05, gamma=0.99,
                     epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                     env_seed=42):
    """
    Train a Q-learning agent on SimpleCrossingS9N1-v0 (no lava) using only (x, y, dir) as state.
    """
    # Create & wrap environment
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = FullyObsWrapper(env)

    # Compute state/action dimensions
    width, height   = env.width, env.height
    num_states      = width * height * 4
    num_actions     = env.action_space.n

    # Initialize Q-table: shape = (324, 3)
    Q_table = np.zeros((num_states, num_actions), dtype=float)

    rewards_list = []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Set seed before each reset to ensure identical environments
        env.seed(env_seed)
        np.random.seed(env_seed)
        
        # Reset environment
        obs = env.reset()

        # Encode initial state
        state_idx = encode_state_agent(env)
        total_reward = 0
        done = False

        # epsilon-greedy action selection
        while not done:
            if np.random.random() < epsilon:
                # Random action
                action = env.action_space.sample()
            else:
                # Greedy action
                action = int(np.argmax(Q_table[state_idx]))

            # Step through the environment, unpacking exactly 4 values
            next_obs, reward, done, info = env.step(action)

            # Encode next state
            next_state_idx = encode_state_agent(env)

            # Q-learning update
            if done:
                # No future reward for terminal states
                td_target = reward  
            else:
                best_next = np.max(Q_table[next_state_idx])
                td_target = reward + gamma * best_next

            td_error = td_target - Q_table[state_idx, action]
            Q_table[state_idx, action] += alpha * td_error

            # Transition and accumulate reward
            state_idx     = next_state_idx
            total_reward += reward

        # Decay epsilon each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_list.append(total_reward)

    return Q_table, rewards_list

if __name__ == "__main__":
    
    num_trials   = 50
    episodes_per_trial = 1000

    print("Q-LEARNING with Deterministic Environment")
    print("=" * 60)
    print("Using seed=42")
    print()

    # Pre‐allocate a NumPy array to store all reward‐sequences
    all_rewards = np.zeros((num_trials, episodes_per_trial), dtype=float)

    for t in range(num_trials):
        print(f"Starting trial {t+1}/{num_trials} ...")
        # Train Q-learning for episodes with FIXED environment
        _, rewards_list = train_q_learning_fixed(
            num_episodes   = episodes_per_trial,
            alpha          = 0.05,
            gamma          = 0.99,
            epsilon_start  = 1.0,
            epsilon_decay  = 0.995,
            epsilon_min    = 0.01,
            env_seed       = 42
        )
        # Save that trial's reward sequence into the array
        all_rewards[t, :] = np.array(rewards_list)
    print("All trials completed.\n")

    # Average reward at each episode index
    avg_reward_per_episode = np.mean(all_rewards, axis=0)

    plt.figure(figsize=(9, 4))
    plt.plot(
        np.arange(1, episodes_per_trial + 1),
        avg_reward_per_episode,
        color="blue", linewidth=2
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Q-Learning: Average Reward per Episode (averaged over {num_trials} trials)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    success_flags = (all_rewards > 0).astype(float)

    block_size   = 100
    num_blocks   = episodes_per_trial // block_size

    trial_success_rate = np.zeros((num_trials, num_blocks), dtype=float)

    for t in range(num_trials):
        trimmed       = success_flags[t, : num_blocks * block_size]
        reshaped      = trimmed.reshape(num_blocks, block_size)
        trial_success_rate[t, :] = reshaped.mean(axis=1)

    # Average across trials
    avg_success_rate_blocks = np.mean(trial_success_rate, axis=0)

    # Build x‐axis
    block_end_episodes = np.arange(1, num_blocks + 1) * block_size

    plt.figure(figsize=(9, 4))
    plt.plot(
        block_end_episodes,
        avg_success_rate_blocks,
        marker="o", linestyle="-", color="green"
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Success Rate (over 100 eps)")
    plt.title(f"FIXED Corrected Q-Learning: Average 100-Episode Success Rate (over {num_trials} trials)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
 
    final_success_rate = avg_success_rate_blocks[-1]
    final_avg_reward = np.mean(avg_reward_per_episode[-100:])
    
    print("Q-LEARNING RESULTS:")
    print("=" * 40)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.3f}")
    print(f"Final average reward (last 100 episodes): {final_avg_reward:.3f}")
    print(f"Peak success rate: {np.max(avg_success_rate_blocks):.3f}")
    print(f"Peak average reward: {np.max(avg_reward_per_episode):.3f}")
    

    print(f"\nConfiguration:")
    print(f"- Environment seed: 42 (FIXED)")
    print(f"- Total episodes per trial: {episodes_per_trial}")
    print(f"- Number of trials: {num_trials}")