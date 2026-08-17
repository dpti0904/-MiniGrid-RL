import gymnasium as gym
import minigrid                                       
from minigrid.wrappers import FullyObsWrapper         
import numpy as np                                    
import matplotlib.pyplot as plt                       

def encode_state_agent(env):
    """
    Convert agent's (x, y, direction) to unique state index.
    Returns: state_idx = dir*(W*H) + y*W + x
    """
    x, y = env.agent_pos       
    d    = env.agent_dir       

    width  = env.width         
    height = env.height        

    return d * (width * height) + y * width + x

def train_sarsa_fixed(num_episodes=1000,
                alpha=0.15, gamma=0.99,
                epsilon_start=1.0, epsilon_decay=0.999, epsilon_min=0.01,
                env_seed=42):
    """
    Train SARSA with fixed environment seed for deterministic experiments.
    """
    # Setup environment
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = FullyObsWrapper(env)  

    # State/action space
    width, height   = env.unwrapped.width, env.unwrapped.height     
    num_states      = width * height * 4        
    num_actions     = env.action_space.n        

    # Initialize Q-table
    Q_table = np.zeros((num_states, num_actions), dtype=float)

    rewards_list = []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Reset with fixed seed for identical environments
        np.random.seed(env_seed)
        obs, _ = env.reset(seed=env_seed)

        # Initial state and action
        state_idx = encode_state_agent(env.unwrapped)
        total_reward = 0
        done = False

        # Pick first action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q_table[state_idx]))

        while not done:
            # Take action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Next state
            next_state_idx = encode_state_agent(env.unwrapped)

            # Pick next action
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.argmax(Q_table[next_state_idx]))

            # SARSA update
            td_target = reward + gamma * Q_table[next_state_idx, next_action]
            td_error = td_target - Q_table[state_idx, action]
            Q_table[state_idx, action] += alpha * td_error

            # Move to next state/action
            state_idx = next_state_idx
            action = next_action
            total_reward += reward

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_list.append(total_reward)

    return Q_table, rewards_list

if __name__ == "__main__":
    # Run 50 trials of 1000 episodes each
    num_trials   = 50
    episodes_per_trial = 1000

    print("FIXED SARSA with Deterministic Environment")
    print("=" * 50)
    print("Using seed=42 for completely identical maze layouts")
    print()

    # Store results from all trials
    all_rewards = np.zeros((num_trials, episodes_per_trial), dtype=float)

    for t in range(num_trials):
        print(f"Starting trial {t+1}/{num_trials} ...")
        _, rewards_list = train_sarsa_fixed(
            num_episodes   = episodes_per_trial,
            alpha          = 0.15,          
            gamma          = 0.99,
            epsilon_start  = 1.0,
            epsilon_decay  = 0.999,        
            epsilon_min    = 0.01,
            env_seed       = 42  
        )
        all_rewards[t, :] = np.array(rewards_list)
    print("All trials completed.\n")

    # Plot average reward per episode
    avg_reward_per_episode = np.mean(all_rewards, axis=0)  

    plt.figure(figsize=(9, 4))
    plt.plot(
        np.arange(1, episodes_per_trial + 1),  
        avg_reward_per_episode,
        color="red", linewidth=2
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"FIXED SARSA: Average Reward per Episode (averaged over {num_trials} trials)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Plot success rate in 100-episode blocks
    success_flags = (all_rewards > 0).astype(float)  

    block_size   = 100
    num_blocks   = episodes_per_trial // block_size  

    trial_success_rate = np.zeros((num_trials, num_blocks), dtype=float)

    for t in range(num_trials):
        trimmed       = success_flags[t, : num_blocks * block_size]
        reshaped      = trimmed.reshape(num_blocks, block_size)  
        trial_success_rate[t, :] = reshaped.mean(axis=1)        

    avg_success_rate_blocks = np.mean(trial_success_rate, axis=0)
    block_end_episodes = np.arange(1, num_blocks + 1) * block_size

    plt.figure(figsize=(9, 4))
    plt.plot(
        block_end_episodes,
        avg_success_rate_blocks,
        marker="o", linestyle="-", color="orange"
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Success Rate (over 100 eps)")
    plt.title(f"FIXED SARSA: Average 100-Episode Success Rate (over {num_trials} trials)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print results
    final_success_rate = avg_success_rate_blocks[-1]
    final_avg_reward = np.mean(avg_reward_per_episode[-500:])  
    
    print("FIXED SARSA RESULTS:")
    print("=" * 30)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.3f}")
    print(f"Final average reward (last 500 episodes): {final_avg_reward:.3f}")
    print(f"Peak success rate: {np.max(avg_success_rate_blocks):.3f}")
    print(f"Peak average reward: {np.max(avg_reward_per_episode):.3f}")
    
    print(f"\nEnvironment Configuration:")
    print(f"- Environment seed: 42 (FIXED)")
    print(f"- Total episodes per trial: {episodes_per_trial}")
    print(f"- Number of trials: {num_trials}")
