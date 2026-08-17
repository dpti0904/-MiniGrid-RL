import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def encode_state_agent(env):
    """Convert agent's (x, y, direction) to unique state index."""
    x, y = env.agent_pos
    d = env.agent_dir
    width, height = env.width, env.height
    return d * (width * height) + y * width + x

def train_algorithm(algorithm_name, num_episodes=1000, env_seed=42):
    """Train either Q-learning or SARSA and return the Q-table."""
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = FullyObsWrapper(env)
    
    width, height = env.unwrapped.width, env.unwrapped.height
    num_states = width * height * 4
    num_actions = env.action_space.n
    
    Q_table = np.zeros((num_states, num_actions), dtype=float)
    
    # Algorithm-specific parameters
    if algorithm_name == "Q-Learning":
        alpha, epsilon_decay = 0.05, 0.995
    else:  # SARSA
        alpha, epsilon_decay = 0.15, 0.999
    
    epsilon = 1.0
    epsilon_min = 0.01
    gamma = 0.99
    
    print(f"Training {algorithm_name}...")
    for ep in range(num_episodes):
        if ep % 200 == 0:
            print(f"  Episode {ep}/{num_episodes}")
            
        np.random.seed(env_seed)
        obs, _ = env.reset(seed=env_seed)
        
        state_idx = encode_state_agent(env.unwrapped)
        done = False
        
        # SARSA: choose initial action
        if algorithm_name == "SARSA":
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q_table[state_idx]))

        while not done:
            # Q-learning: choose action here
            if algorithm_name == "Q-Learning":
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(Q_table[state_idx]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state_idx = encode_state_agent(env.unwrapped)

            if algorithm_name == "Q-Learning":
                # Q-learning update
                if done:
                    td_target = reward
                else:
                    best_next = np.max(Q_table[next_state_idx])
                    td_target = reward + gamma * best_next
            else:
                # SARSA: choose next action
                if np.random.random() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = int(np.argmax(Q_table[next_state_idx]))
                
                # SARSA update
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * Q_table[next_state_idx, next_action]

            td_error = td_target - Q_table[state_idx, action]
            Q_table[state_idx, action] += alpha * td_error

            state_idx = next_state_idx
            if algorithm_name == "SARSA":
                action = next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print(f"{algorithm_name} training completed!")
    return Q_table, env

def run_episode_with_policy(Q_table, env, max_steps=50):
    """Run one episode using the learned policy and record the path."""
    np.random.seed(42)  # Fixed seed for consistent starting position
    obs, _ = env.reset(seed=42)
    
    path = []
    state_idx = encode_state_agent(env.unwrapped)
    done = False
    steps = 0
    
    # Record starting position
    x, y = env.unwrapped.agent_pos
    path.append((x, y))
    
    while not done and steps < max_steps:
        # Use greedy policy (no exploration)
        action = int(np.argmax(Q_table[state_idx]))
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record new position
        x, y = env.unwrapped.agent_pos
        path.append((x, y))
        
        state_idx = encode_state_agent(env.unwrapped)
        steps += 1
    
    return path, reward > 0, steps

def get_environment_layout(env):
    """Get the environment layout for visualization."""
    np.random.seed(42)
    env.reset(seed=42)
    grid = env.unwrapped.grid
    width, height = grid.width, grid.height
    
    layout = np.zeros((height, width))
    start_pos = None
    goal_pos = None
    
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None:
                if cell.type == 'wall':
                    layout[y, x] = 1
                elif cell.type == 'goal':
                    layout[y, x] = 2
                    goal_pos = (x, y)
    
    start_pos = tuple(env.unwrapped.agent_pos)
    
    return layout, start_pos, goal_pos

def visualize_paths(q_path, sarsa_path, q_success, sarsa_success, q_steps, sarsa_steps, env):
    """Create a simple visualization showing the paths taken by both algorithms."""
    layout, start_pos, goal_pos = get_environment_layout(env)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color scheme
    colors = ['white', 'black', 'gold']  # empty, wall, goal
    cmap = ListedColormap(colors)
    
    # Q-Learning Path
    ax1 = axes[0]
    ax1.imshow(layout, cmap=cmap, vmin=0, vmax=2)
    ax1.set_title(f'Q-Learning Path\n{"SUCCESS" if q_success else "FAILED"} in {q_steps} steps', 
                  fontsize=14, fontweight='bold')
    
    # Draw path
    if len(q_path) > 1:
        path_x = [pos[0] for pos in q_path]
        path_y = [pos[1] for pos in q_path]
        ax1.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, label='Path')
        
        # Mark steps with numbers
        for i, (x, y) in enumerate(q_path[:-1]):  # Don't number the last position
            ax1.text(x, y, str(i+1), fontsize=10, ha='center', va='center', 
                    bbox=dict(boxstyle='circle', facecolor='lightblue', alpha=0.8))
    
    # Mark start and goal
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    ax1.legend()
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # SARSA Path
    ax2 = axes[1]
    ax2.imshow(layout, cmap=cmap, vmin=0, vmax=2)
    ax2.set_title(f'SARSA Path\n{"SUCCESS" if sarsa_success else "FAILED"} in {sarsa_steps} steps', 
                  fontsize=14, fontweight='bold')
    
    # Draw path
    if len(sarsa_path) > 1:
        path_x = [pos[0] for pos in sarsa_path]
        path_y = [pos[1] for pos in sarsa_path]
        ax2.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, label='Path')
        
        # Mark steps with numbers
        for i, (x, y) in enumerate(sarsa_path[:-1]):
            ax2.text(x, y, str(i+1), fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='lightcoral', alpha=0.8))
    
    # Mark start and goal
    ax2.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax2.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    ax2.legend()
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # 3. Path Comparison
    ax3 = axes[2]
    ax3.imshow(layout, cmap=cmap, vmin=0, vmax=2)
    ax3.set_title('Path Comparison\nBlue: Q-Learning, Red: SARSA', fontsize=14, fontweight='bold')
    
    # Draw both paths
    if len(q_path) > 1:
        path_x = [pos[0] for pos in q_path]
        path_y = [pos[1] for pos in q_path]
        ax3.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, label=f'Q-Learning ({q_steps} steps)')
    
    if len(sarsa_path) > 1:
        path_x = [pos[0] for pos in sarsa_path]
        path_y = [pos[1] for pos in sarsa_path]
        ax3.plot(path_x, path_y, 'r--', linewidth=3, alpha=0.7, label=f'SARSA ({sarsa_steps} steps)')
    
    # Mark start and goal
    ax3.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start')
    ax3.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')
    ax3.legend()
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    
    # Add grid lines to all plots
    for ax in axes:
        ax.set_xticks(range(env.unwrapped.width))
        ax.set_yticks(range(env.unwrapped.height))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_path_analysis(q_path, sarsa_path, q_success, sarsa_success, q_steps, sarsa_steps):
    """Print detailed path analysis."""
    print("\n" + "="*60)
    print("PATH ANALYSIS")
    print("="*60)
    
    print(f"Q-Learning:")
    print(f"  Result: {'SUCCESS' if q_success else 'FAILED'}")
    print(f"  Steps taken: {q_steps}")
    print(f"  Path efficiency: {len(q_path)-1} position changes")
    
    print(f"\nSARSA:")
    print(f"  Result: {'SUCCESS' if sarsa_success else 'FAILED'}")
    print(f"  Steps taken: {sarsa_steps}")
    print(f"  Path efficiency: {len(sarsa_path)-1} position changes")
    
    if q_success and sarsa_success:
        print(f"\nComparison:")
        if q_steps < sarsa_steps:
            print(f"  Q-Learning was more efficient by {sarsa_steps - q_steps} steps")
        elif sarsa_steps < q_steps:
            print(f"  SARSA was more efficient by {q_steps - sarsa_steps} steps")
        else:
            print(f"  Both algorithms took the same number of steps")
    
    print(f"\nDetailed Paths:")
    print(f"Q-Learning path: {' -> '.join([f'({x},{y})' for x, y in q_path])}")
    print(f"SARSA path: {' -> '.join([f'({x},{y})' for x, y in sarsa_path])}")

if __name__ == "__main__":
    print("SIMPLE PATH VISUALIZATION")
    print("=" * 50)
    print("Training Q-learning and SARSA, then showing the paths they take")
    print()
    
    # Train both algorithms
    q_table, env = train_algorithm("Q-Learning")
    sarsa_table, _ = train_algorithm("SARSA")
    
    print("\nRunning episodes with learned policies...")
    
    # Run episodes with learned policies
    q_path, q_success, q_steps = run_episode_with_policy(q_table, env)
    sarsa_path, sarsa_success, sarsa_steps = run_episode_with_policy(sarsa_table, env)
    
    # Visualize the paths
    print("\nCreating path visualization...")
    visualize_paths(q_path, sarsa_path, q_success, sarsa_success, q_steps, sarsa_steps, env)
    
    # Print analysis
    print_path_analysis(q_path, sarsa_path, q_success, sarsa_success, q_steps, sarsa_steps)
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETE!")