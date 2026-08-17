import numpy as np
import matplotlib.pyplot as plt
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import FullyObsWrapper

class RandCrossingEnv(MiniGridEnv):
    """
    A crossing environment that, on every reset(), rebuilds:
      1. A random interior "river" row and a random opening column
      2. A random agent start position (from free cells)
      3. A random goal position (from free cells, excluding agent start)

    This allows us to call reset() multiple times on the same instance and verify
    that the wall layout, agent start, and goal change each time.
    """
    def __init__(self, width=9, height=9, max_steps=100, render_mode=None):
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            render_mode=render_mode
        )

    def _gen_grid(self, width, height):
        # Create empty grid with outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Pick a random interior row for the "river" and a random opening column
        r = np.random.randint(2, height - 2)
        opening_col = np.random.randint(1, width - 1)
        for c in range(1, width - 1):
            if c == opening_col:
                continue
            self.grid.set(c, r, Wall())

        # Gather free cells on each side of the wall
        cells_above_wall = []
        cells_below_wall = []
        
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:
                    if y < r:
                        cells_above_wall.append((x, y))
                    elif y > r:
                        cells_below_wall.append((x, y))
                    # Skip cells at y == r (the wall row)

        # Ensure we have cells on both sides
        if len(cells_above_wall) == 0 or len(cells_below_wall) == 0:
            # Fallback: if one side is empty, place cells near the opening
            if len(cells_above_wall) == 0 and r > 1:
                cells_above_wall.append((opening_col, r - 1))
            if len(cells_below_wall) == 0 and r < height - 2:
                cells_below_wall.append((opening_col, r + 1))

        # Randomly choose which side gets the agent (50/50 chance)
        if np.random.random() < 0.5:
            agent_cells = cells_above_wall
            goal_cells = cells_below_wall
            agent_side = "above"
            goal_side = "below"
        else:
            agent_cells = cells_below_wall
            goal_cells = cells_above_wall
            agent_side = "below"
            goal_side = "above"

        # Place agent on chosen side
        self.agent_pos = agent_cells[np.random.randint(len(agent_cells))]
        self.agent_dir = 0

        # Place goal on opposite side
        goal_pos = goal_cells[np.random.randint(len(goal_cells))]

        # Place the Goal object
        self.put_obj(Goal(), *goal_pos)

        # Set mission
        self.mission = "reach the goal"
        
        # Store information for debugging
        self._wall_row = r
        self._opening_col = opening_col
        self._agent_side = agent_side
        self._goal_side = goal_side

    def step(self, action):
        return super().step(action)

def encode_state_agent(env):
    """
    Encode the agent's (x, y, direction) into a unique integer.
    """
    x, y = env.agent_pos
    d    = env.agent_dir

    width  = env.width
    height = env.height

    return d * (width * height) + y * width + x

def train_q_learning_randomized(num_episodes=1000,
                     alpha=0.05, gamma=0.99,
                     epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    """
    Train a Q-learning agent on randomized RandCrossingEnv using only (x, y, dir) as state.
    Each episode has different agent start, goal position, and wall layout.
    Properly handles terminal states (no future reward when done=True).
    """
    # Create & wrap environment
    env = RandCrossingEnv()
    env = FullyObsWrapper(env)

    # Compute state/action dimensions
    width, height   = env.unwrapped.width, env.unwrapped.height
    num_states      = width * height * 4
    num_actions     = env.action_space.n

    # Initialize Q-table
    Q_table = np.zeros((num_states, num_actions), dtype=float)

    rewards_list = []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Each reset creates a new random environment layout
        obs, info = env.reset()

        # Encode initial state
        state_idx = encode_state_agent(env.unwrapped)
        total_reward = 0
        done = False

        while not done:
            # epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q_table[state_idx]))

            # Step through the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Encode next state
            next_state_idx = encode_state_agent(env.unwrapped)

            # Q-learning update
            if done:
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
    # Run 50 trials, each with 1000 episodes, collecting their episode‐returns.
    num_trials   = 50
    episodes_per_trial = 1000

    print("RANDOMIZED Q-LEARNING with Domain Randomization")
    print("=" * 60)
    print("EVERY episode has randomized:")
    print("- Agent starting position")
    print("- Goal position") 
    print("- Wall layout (river position and opening)")
    print("- Agent and goal are ALWAYS on opposite sides of the wall")
    print("Proper terminal state handling (no future reward when done=True)")
    print()

    # Pre‐allocate a NumPy array to store all reward‐sequences
    all_rewards = np.zeros((num_trials, episodes_per_trial), dtype=float)

    for t in range(num_trials):
        print(f"Starting trial {t+1}/{num_trials} ...")
        # Train Q-learning for episodes with RANDOMIZED environment
        _, rewards_list = train_q_learning_randomized(
            num_episodes   = episodes_per_trial,
            alpha          = 0.05,
            gamma          = 0.99,
            epsilon_start  = 1.0,
            epsilon_decay  = 0.995,
            epsilon_min    = 0.01
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
        color="red", linewidth=2
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"RANDOMIZED Q-Learning: Average Reward per Episode (averaged over {num_trials} trials)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Compute and plot the *average 100‐episode success rate* across all trials.
    success_flags = (all_rewards > 0).astype(float)

    block_size   = 100
    num_blocks   = episodes_per_trial // block_size

    # We will compute, for each trial t, its success‐rate in each 100‐episode block:
    trial_success_rate = np.zeros((num_trials, num_blocks), dtype=float)

    for t in range(num_trials):
        # Take trial t's success flags, reshape into (num_blocks, block_size)
        trimmed       = success_flags[t, : num_blocks * block_size]
        reshaped      = trimmed.reshape(num_blocks, block_size)
        trial_success_rate[t, :] = reshaped.mean(axis=1)

    # Now average across trials: shape = (num_blocks,)
    avg_success_rate_blocks = np.mean(trial_success_rate, axis=0)

    # Build x‐axis
    block_end_episodes = np.arange(1, num_blocks + 1) * block_size

    plt.figure(figsize=(9, 4))
    plt.plot(
        block_end_episodes,
        avg_success_rate_blocks,
        marker="o", linestyle="-", color="orange"
    )
    plt.xlabel("Episode")
    plt.ylabel("Average Success Rate (over 100 eps)")
    plt.title(f"RANDOMIZED Q-Learning: Average 100-Episode Success Rate (over {num_trials} trials)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    final_success_rate = avg_success_rate_blocks[-1]
    final_avg_reward = np.mean(avg_reward_per_episode[-100:])
    
    print("RANDOMIZED Q-LEARNING RESULTS:")
    print("=" * 40)
    print(f"Final success rate (last 100 episodes): {final_success_rate:.3f}")
    print(f"Final average reward (last 100 episodes): {final_avg_reward:.3f}")
    print(f"Peak success rate: {np.max(avg_success_rate_blocks):.3f}")
    print(f"Peak average reward: {np.max(avg_reward_per_episode):.3f}")
    
    print(f"\nEnvironment Configuration:")
    print(f"- Environment: FULLY RANDOMIZED every episode")
    print(f"- Total episodes per trial: {episodes_per_trial}")
    print(f"- Number of trials: {num_trials}")
    
    print(f"\nAlgorithm Details:")
    print(f"- Algorithm: Q-Learning (off-policy)")
    print(f"- Learning rate (alpha): 0.05")
    print(f"- Discount factor (gamma): 0.99")
    print(f"- Epsilon decay: 0.9998")