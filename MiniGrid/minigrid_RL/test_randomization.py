import numpy as np
import time
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
        cells_above_wall = []  # y < r (above the wall)
        cells_below_wall = []  # y > r (below the wall)
        
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:  # Free cell
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

def test_environment_randomization():
    """
    Test the randomization in custom RandCrossingEnv by creating 5 episodes
    and visualizing the differences in agent start position, goal position, and crossings.
    """
    
    print("Testing Custom RandCrossingEnv Domain Randomization")
    print("=" * 50)
    print("This tests FULL randomization: agent position, goal position, AND wall layout!")
    print()
    
    # Test 5 different episodes
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Create a fresh environment for each episode with human render mode
        env = RandCrossingEnv(render_mode="human")
        env = FullyObsWrapper(env)
        
        # Reset environment (this should randomize positions AND layout)
        obs, info = env.reset()
        
        # Get agent and goal positions from the unwrapped environment
        agent_pos = env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir
        
        # Find goal position by looking through the grid
        goal_pos = None
        for y in range(env.unwrapped.height):
            for x in range(env.unwrapped.width):
                cell = env.unwrapped.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        # Print randomization details
        print(f"  Agent position: {agent_pos} (side: {env.unwrapped._agent_side}), direction: {agent_dir}")
        print(f"  Goal position: {goal_pos} (side: {env.unwrapped._goal_side})")
        print(f"  Wall row: {env.unwrapped._wall_row}, opening column: {env.unwrapped._opening_col}")
        print(f"  Agent and goal on OPPOSITE sides: ✓")
        
        # Print a simple text representation of the grid
        print("  Grid layout:")
        for y in range(env.unwrapped.height):
            row = "    "
            for x in range(env.unwrapped.width):
                if (x, y) == agent_pos:
                    row += "A "  # Agent
                elif (x, y) == goal_pos:
                    row += "G "  # Goal
                else:
                    cell = env.unwrapped.grid.get(x, y)
                    if cell is None:
                        row += ". "  # Empty
                    elif cell.type == "wall":
                        row += "# "  # Wall
                    else:
                        row += "? "  # Other
            print(row)
        
        # Try to render the environment
        try:
            print("  Opening visual window...")
            # Force rendering with proper setup
            env.unwrapped.render()
            print("Visual window opened! Check for pygame window.")
            
            # Keep window open for user to see
            print("  Window will stay open for 3 seconds...")
            time.sleep(3)
            
        except Exception as e:
            print(f"Visual rendering failed: {e}")
            print("  This might be due to display/pygame setup issues.")
            print("  But you can see the text grid above.")
        
        # Wait for user input
        print("  Press Enter to continue to next episode...")
        input()
        
        # Close the environment for this episode
        try:
            env.close()
        except:
            pass
    
    print("\nTest complete! You should notice:")    

def test_environment_positions_only():
    """
    Alternative test that just prints positions without rendering
    (in case rendering doesn't work properly)
    """
    env = RandCrossingEnv()
    env = FullyObsWrapper(env)
    
    print("Testing DOMAIN RANDOMIZATION across 10 resets:")
    print("=" * 50)
    print("This includes agent position, goal position, AND wall layout randomization")
    print()
    
    configurations = []
    
    for i in range(10):
        obs, info = env.reset()
        agent_pos = env.unwrapped.agent_pos
        
        # Find goal position
        goal_pos = None
        for y in range(env.unwrapped.height):
            for x in range(env.unwrapped.width):
                cell = env.unwrapped.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        wall_row = env.unwrapped._wall_row
        opening_col = env.unwrapped._opening_col
        agent_side = env.unwrapped._agent_side
        goal_side = env.unwrapped._goal_side
        
        config = (agent_pos, goal_pos, wall_row, opening_col, agent_side, goal_side)
        configurations.append(config)
        
        print(f"Reset {i+1:2d}: Agent={agent_pos} ({agent_side:5s}), Goal={goal_pos} ({goal_side:5s}), "
              f"Wall_row={wall_row}, Opening_col={opening_col}")
    
    # Check if all configurations are the same
    all_same = all(config == configurations[0] for config in configurations)
    print(f"\nAre all configurations identical? {all_same}")
    
    if not all_same:
        print("Environment IS FULLY randomized - positions AND layouts change!")
        
        # Count unique configurations
        unique_configs = len(set(configurations))
        print(f"Found {unique_configs} unique configurations out of 10 resets")
        
        # Check specific randomization aspects
        agent_positions = [config[0] for config in configurations]
        goal_positions = [config[1] for config in configurations]
        wall_rows = [config[2] for config in configurations]
        opening_cols = [config[3] for config in configurations]
        
        unique_agent_pos = len(set(agent_positions))
        unique_goal_pos = len(set(goal_positions))
        unique_wall_rows = len(set(wall_rows))
        unique_opening_cols = len(set(opening_cols))
        
        print(f"Unique agent positions: {unique_agent_pos}")
        print(f"Unique goal positions: {unique_goal_pos}")
        print(f"Unique wall rows: {unique_wall_rows}")
        print(f"Unique opening columns: {unique_opening_cols}")
        
    else:
        print("Environment appears to be deterministic - all configurations identical")
    
    env.close()

def test_with_text_visualization():
    """
    Show 5 different episodes with text-based grid visualization
    """
    print("Testing DOMAIN RANDOMIZATION with text visualization - 5 episodes:")
    print("=" * 60)
    print("Legend: A=Agent, G=Goal, #=Wall, .=Empty")
    print("RANDOMIZED: Agent pos, Goal pos, Wall layout, Opening position")
    print()
    
    for episode in range(5):
        # Create fresh environment for each test
        env = RandCrossingEnv()
        env = FullyObsWrapper(env)
        
        obs, info = env.reset()
        agent_pos = env.unwrapped.agent_pos
        
        # Find goal position
        goal_pos = None
        for y in range(env.unwrapped.height):
            for x in range(env.unwrapped.width):
                cell = env.unwrapped.grid.get(x, y)
                if cell and cell.type == "goal":
                    goal_pos = (x, y)
                    break
            if goal_pos:
                break
        
        print(f"Episode {episode + 1}:")
        print(f"  Agent: {agent_pos} ({env.unwrapped._agent_side} side), Goal: {goal_pos} ({env.unwrapped._goal_side} side)")
        print(f"  Wall row: {env.unwrapped._wall_row}, Opening column: {env.unwrapped._opening_col}")
        
        # Print grid
        for y in range(env.unwrapped.height):
            row = "  "
            for x in range(env.unwrapped.width):
                if (x, y) == agent_pos:
                    row += "A "
                elif (x, y) == goal_pos:
                    row += "G "
                else:
                    cell = env.unwrapped.grid.get(x, y)
                    if cell is None:
                        row += ". "
                    elif cell.type == "wall":
                        row += "# "
                    else:
                        row += "? "
            print(row)
        print()
        
        env.close()

def test_matplotlib_visualization():
    """
    Alternative visualization using matplotlib (in case pygame doesn't work)
    """
    print("Testing DOMAIN RANDOMIZATION with matplotlib visualization:")
    print("=" * 60)
    
    # Create 2x3 subplot grid for 5 episodes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for episode in range(5):
        env = RandCrossingEnv()
        env = FullyObsWrapper(env)
        
        obs, info = env.reset()
        
        # Get the RGB array representation
        try:
            # Try to get visual representation
            rgb_array = env.unwrapped.get_frame()
            
            # Display in subplot
            ax = axes[episode]
            ax.imshow(rgb_array)
            ax.set_title(f"Episode {episode + 1}\n"
                        f"Agent: {env.unwrapped.agent_pos} ({env.unwrapped._agent_side})\n"
                        f"Wall row: {env.unwrapped._wall_row}, Opening: {env.unwrapped._opening_col}")
            ax.axis('off')
            
        except Exception as e:
            # Fallback to text representation if visual fails
            ax = axes[episode]
            ax.text(0.5, 0.5, f"Episode {episode + 1}\n"
                              f"Agent: {env.unwrapped.agent_pos}\n"
                              f"Goal: Find in grid\n"
                              f"Wall row: {env.unwrapped._wall_row}\n"
                              f"Opening: {env.unwrapped._opening_col}\n"
                              f"Visual failed: {str(e)[:30]}...",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            ax.axis('off')
        
        env.close()
    
    # Hide the last subplot since we only have 5 episodes
    axes[5].axis('off')
    
    plt.suptitle("Domain Randomization Test - Multiple Episodes", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("Matplotlib visualization complete!")

if __name__ == "__main__":
    print("Choose DOMAIN RANDOMIZATION test mode:")
    print("1. Visual test (tries to open pygame windows)")
    print("2. Position-only test (console output)")
    print("3. Text visualization test (shows grid layouts)")
    print("4. Matplotlib visualization test (plots in Python)")
    
    choice = input("Enter choice (1, 2, 3, or 4): ").strip()
    
    if choice == "1":
        test_environment_randomization()
    elif choice == "2":
        test_environment_positions_only()
    elif choice == "3":
        test_with_text_visualization()
    elif choice == "4":
        test_matplotlib_visualization()
    else:
        print("Invalid choice. Running text visualization test...")
        test_with_text_visualization() 