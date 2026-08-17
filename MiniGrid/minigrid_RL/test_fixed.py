import gym
import gym_minigrid
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np

def test_fixed_environment_visual():
    """
    Test with a fixed seed to ensure agent position, goal position, 
    and maze layout are identical across all episodes
    """
    print("Testing SimpleCrossingS9N1-v0 with FIXED configuration:")
    print("=" * 60)
    print("Using seed=42 to force identical layouts\n")
    
    all_positions = []
    all_grids = []
    
    for i in range(5):
        print(f"\nEpisode {i+1}:")
        
        # Create environment with fixed seed for each reset
        env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
        env = FullyObsWrapper(env)
        
        # Set the same seed every time to get identical configurations
        env.seed(40)
        np.random.seed(40)
        
        obs = env.reset()
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        
        # Find goal position
        goal_pos = None
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        # Record the grid layout (focusing on walls)
        grid_layout = []
        for y in range(env.height):
            row = []
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell:
                    row.append(cell.type[0])  # First letter of type
                else:
                    row.append(' ')  # Empty space
            grid_layout.append(''.join(row))
        
        all_positions.append((agent_pos, goal_pos, agent_dir))
        all_grids.append(grid_layout)
        
        print(f"  Agent={agent_pos}, Goal={goal_pos}, Dir={agent_dir}")
        
        # Print the visual grid layout for this episode
        print("  Text Grid layout:")
        for y in range(env.height):
            row = "    "
            for x in range(env.width):
                if (x, y) == agent_pos:
                    row += "A "  # Agent
                elif (x, y) == goal_pos:
                    row += "G "  # Goal
                else:
                    cell = env.grid.get(x, y)
                    if cell is None:
                        row += ". "  # Empty
                    elif cell.type == "wall":
                        row += "# "  # Wall
                    else:
                        row += "? "  # Other
            print(row)
        
        # Render visual window
        try:
            # Try multiple render calls to ensure it works
            img = env.render(mode='rgb_array')
            env.render(mode='human')
            print("Visual window should have opened. Check for a new window!")
        except Exception as e:
            print(f"Visual rendering failed: {e}")
            print("But you can see the text grid above.")
        
        # Wait for user input
        print("  Press Enter to continue to next episode...")
        input()
        
        env.close()
    
    # Check if everything is identical
    unique_positions = set(all_positions)
    unique_grids = set(tuple(grid) for grid in all_grids)
    
    print(f"\nSummary after 5 episodes with fixed seed:")
    print(f"Unique position combinations: {len(unique_positions)}")
    print(f"Unique grid layouts: {len(unique_grids)}")
    
    if len(unique_positions) == 1 and len(unique_grids) == 1:
        print("SUCCESS: Environment is now DETERMINISTIC")
    else:
        print("FAILED: Environment still has randomization")
        if len(unique_positions) > 1:
            print(f"  - Found {len(unique_positions)} different position combinations")
        if len(unique_grids) > 1:
            print(f"  - Found {len(unique_grids)} different grid layouts")

def test_fixed_environment():
    """
    Test with a fixed seed to ensure agent position, goal position, 
    and maze layout are identical across all episodes
    """
    print("Testing SimpleCrossingS9N1-v0 with FIXED configuration:")
    print("=" * 60)
    print("Using seed=42 to force identical layouts\n")
    
    all_positions = []
    all_grids = []
    
    for i in range(5):
        # Create environment with fixed seed for each reset
        env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
        env = FullyObsWrapper(env)
        
        # Set the same seed every time to get identical configurations
        env.seed(42)
        np.random.seed(42)
        
        obs = env.reset()
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        
        # Find goal position
        goal_pos = None
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        # Record the grid layout (focusing on walls)
        grid_layout = []
        for y in range(env.height):
            row = []
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell:
                    row.append(cell.type[0])  # First letter of type
                else:
                    row.append(' ')  # Empty space
            grid_layout.append(''.join(row))
        
        all_positions.append((agent_pos, goal_pos, agent_dir))
        all_grids.append(grid_layout)
        
        print(f"Episode {i+1}: Agent={agent_pos}, Goal={goal_pos}, Dir={agent_dir}")
        
        # Print the visual grid layout for this episode
        print("  Grid layout:")
        for y in range(env.height):
            row = "    "
            for x in range(env.width):
                if (x, y) == agent_pos:
                    row += "A "  # Agent
                elif (x, y) == goal_pos:
                    row += "G "  # Goal
                else:
                    cell = env.grid.get(x, y)
                    if cell is None:
                        row += ". "  # Empty
                    elif cell.type == "wall":
                        row += "# "  # Wall
                    else:
                        row += "? "  # Other
            print(row)
        print()
        
        env.close()
    
    # Check if everything is identical
    unique_positions = set(all_positions)
    unique_grids = set(tuple(grid) for grid in all_grids)
    
    print(f"Summary after 5 episodes with fixed seed:")
    print(f"Unique position combinations: {len(unique_positions)}")
    print(f"Unique grid layouts: {len(unique_grids)}")
    
    if len(unique_positions) == 1 and len(unique_grids) == 1:
        print("SUCCESS: Environment is now DETERMINISTIC")
    else:
        print("FAILED: Environment still has randomization")
        if len(unique_positions) > 1:
            print(f"  - Found {len(unique_positions)} different position combinations")
        if len(unique_grids) > 1:
            print(f"  - Found {len(unique_grids)} different grid layouts")

def test_comparison():
    """
    Compare the original randomized version vs fixed version
    """
    print("\n" + "="*60)
    print("COMPARISON: Random vs Fixed Environment")
    print("="*60)
    
    # Test random version
    print("\n1. RANDOM VERSION (no seed):")
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = FullyObsWrapper(env)
    
    random_positions = []
    for i in range(3):
        obs = env.reset()
        agent_pos = env.agent_pos
        
        goal_pos = None
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        random_positions.append((agent_pos, goal_pos))
        print(f"  Reset {i+1}: Agent={agent_pos}, Goal={goal_pos}")
    
    env.close()
    
    # Test fixed version
    print("\n2. FIXED VERSION (with seed=42):")
    fixed_positions = []
    for i in range(3):
        env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
        env = FullyObsWrapper(env)
        env.seed(42)
        np.random.seed(42)
        
        obs = env.reset()
        agent_pos = env.agent_pos
        
        goal_pos = None
        for y in range(env.height):
            for x in range(env.width):
                cell = env.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        fixed_positions.append((agent_pos, goal_pos))
        print(f"  Reset {i+1}: Agent={agent_pos}, Goal={goal_pos}")
        env.close()
    
    print(f"\nRandom version unique configs: {len(set(random_positions))}")
    print(f"Fixed version unique configs: {len(set(fixed_positions))}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Fixed environment with VISUAL WINDOWS")
    print("2. Fixed environment text-only")
    print("3. Comparison (random vs fixed)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_fixed_environment_visual()
    elif choice == "2":
        test_fixed_environment()
    elif choice == "3":
        test_comparison()
    else:
        print("Invalid choice. Running visual test...")
        test_fixed_environment_visual() 