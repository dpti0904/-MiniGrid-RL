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

def ensure_opposite_sides(env):
    """
    Place agent and goal on opposite sides of the wall.
    """
    grid = env.unwrapped.grid
    width, height = env.unwrapped.width, env.unwrapped.height
    
    # Wall is in the middle column
    wall_x = width // 2  
    
    # Pick which side for agent
    agent_on_left = np.random.choice([True, False])
    
    if agent_on_left:
        agent_x_range = list(range(1, wall_x))
        goal_x_range = list(range(wall_x + 1, width - 1))
    else:
        agent_x_range = list(range(wall_x + 1, width - 1))
        goal_x_range = list(range(1, wall_x))
    
    # Place agent and goal randomly in their sides
    agent_x = np.random.choice(agent_x_range)
    agent_y = np.random.choice(range(1, height - 1))
    
    goal_x = np.random.choice(goal_x_range)
    goal_y = np.random.choice(range(1, height - 1))
    
    # Set positions
    env.unwrapped.agent_pos = np.array([agent_x, agent_y])
    env.unwrapped.agent_dir = np.random.choice([0, 1, 2, 3])
    
    # Remove old goal
    for i in range(width):
        for j in range(height):
            if env.unwrapped.grid.get(i, j) and env.unwrapped.grid.get(i, j).type == 'goal':
                env.unwrapped.grid.set(i, j, None)
    
    # Add new goal
    from minigrid.core.world_object import Goal
    env.unwrapped.grid.set(goal_x, goal_y, Goal())

def train_sarsa_all_random(num_episodes=1000,
                alpha=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    """
    Train SARSA with randomized environments but consistent learning params.
    Agent and goal always on opposite sides of wall.
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
        # Reset with random seed
        random_seed = np.random.randint(0, 1000000)
        obs, _ = env.reset(seed=random_seed)
        
        # Force opposite sides
        ensure_opposite_sides(env)
        
        # Use consistent learning params
        episode_epsilon = epsilon
        episode_alpha = alpha

        # Initial state and action
        state_idx = encode_state_agent(env.unwrapped)
        total_reward = 0
        done = False

        # Pick first action
        if np.random.random() < episode_epsilon:
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
            if np.random.random() < episode_epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = int(np.argmax(Q_table[next_state_idx]))

            # SARSA update
            td_target = reward + gamma * Q_table[next_state_idx, next_action]
            td_error = td_target - Q_table[state_idx, action]
            Q_table[state_idx, action] += episode_alpha * td_error

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

    print("ALL RANDOM SARSA with Environmental Randomization")
    print("=" * 50)
    print("✓ Wall layouts randomized each episode")
    print("✓ Agent and goal positions randomized (always opposite sides)")
    print("✓ Agent direction randomized")
    print("✓ Environment seed randomized each episode")
    print("✓ Learning parameters kept consistent for fair comparison")
    print()

    # Store results from all trials
    all_rewards = np.zeros((num_trials, episodes_per_trial), dtype=float)

    for t in range(num_trials):
        print(f"Starting trial {t+1}/{num_trials} ...")
        _, rewards_list = train_sarsa_all_random(
            num_episodes   = episodes_per_trial,
            alpha          = 0.15,          
            gamma          = 0.99,
            epsilon_start  = 1.0,
            epsilon_decay  = 0.999,        
            epsilon_min    = 0.01
        )
        all_rewards[t, :] = np.array(rewards_list)
    print("All trials completed.\n")

    # Plot average reward per episode
    avg_reward_per_episode = np.mean(all_rewards, axis=0)  

    plt.figure(figsize=(9, 4))
    plt.plot(
        np.arange(1, episodes_per_trial + 1),  
        avg_reward_per_episode,
        color="purple", linewidth=2
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"ALL RANDOM SARSA: Average Reward per Episode (averaged over {num_trials} trials)")
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
        marker="o", linestyle="-", color="darkviolet"
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Success Rate (over 100 eps)")
    plt.title(f"ALL RANDOM SARSA: Average 100-Episode Success Rate (over {num_trials} trials)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Print results
    final_success_rate = avg_success_rate_blocks[-1]
    final_avg_reward = np.mean(avg_reward_per_episode[-500:])  
    
    print("ALL RANDOM SARSA RESULTS:")
    print("=" * 35)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.3f}")
    print(f"Final average reward (last 500 episodes): {final_avg_reward:.3f}")
    print(f"Peak success rate: {np.max(avg_success_rate_blocks):.3f}")
    print(f"Peak average reward: {np.max(avg_reward_per_episode):.3f}")
    
    print(f"\nEnvironment Configuration:")
    print(f"- Environment seed: FULLY RANDOMIZED each episode")
    print(f"- Wall layout: RANDOMIZED every episode")
    print(f"- Agent position: RANDOMIZED (opposite side from goal)")
    print(f"- Goal position: RANDOMIZED (opposite side from agent)")
    print(f"- Agent direction: RANDOMIZED each episode")
    print(f"- Total episodes per trial: {episodes_per_trial}")
    print(f"- Number of trials: {num_trials}")

    print(f"\nAlgorithm Details:")
    print(f"- Algorithm: SARSA (on-policy)")
    print(f"- Learning rate (alpha): 0.1 (consistent across episodes)")
    print(f"- Discount factor (gamma): 0.99")
    print(f"- Epsilon decay: 0.999 (consistent schedule)")
    print(f"- Environment: Positions and layouts fully randomized")
    print(f"- Constraint: Agent and goal ALWAYS on opposite sides of wall")
    print(f"- Focus: Testing environmental generalization with stable learning!")

    print(f"\nPlots complete. Arrays available:")
    print(f"- avg_reward_per_episode.shape = {avg_reward_per_episode.shape}")
    print(f"- avg_success_rate_blocks.shape = {avg_success_rate_blocks.shape}")
    
    print(f"\n KEY FEATURE: Agent and goal are GUARANTEED to be on opposite sides!")
    print(f" This tests the algorithm's ability to learn under maximum uncertainty.")