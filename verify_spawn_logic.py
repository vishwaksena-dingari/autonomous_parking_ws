import gymnasium as gym
import matplotlib.pyplot as plt
import os
import sys

# Add the source directory to the path so we can import the module
sys.path.append(os.path.join(os.getcwd(), "src/autonomous_parking"))

from autonomous_parking.env2d.parking_env import ParkingEnv

import random

def verify_spawns():
    # Create output directory
    output_dir = "spawn_verification"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old images
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    
    levels = [
        (0, "Level 1 (Easy) - On Road, Aligned"),
        (600, "Level 2 (Medium) - On Road, Random Yaw"),
        (1200, "Level 3 (Hard) - Entrance")
    ]
    
    for episode_count, description in levels:
        print(f"\n{'='*60}")
        print(f"Testing {description}...")
        print(f"{'='*60}")
        
        # --------------------------------------------------------------
        # Force a mix of H-bays (horizontal road) and V-bays (vertical road)
        # --------------------------------------------------------------
        for i in range(20):  # 20 samples per level
            # Alternate: even indices → H-bay, odd indices → V-bay
            forced_type = "H" if i % 2 == 0 else "V"
            
            # Choose a random lot (lot_a or lot_b)
            lot_name = random.choice(["lot_a", "lot_b"])
            env = ParkingEnv(render_mode="rgb_array", lot_name=lot_name)
            env.episode_count = episode_count
            
            # Pick a goal bay of the forced type (if any exist)
            candidate_bays = [b for b in env.bays if b["id"].upper().startswith(forced_type)]
            if not candidate_bays:  # fallback – just use any bay
                bay_id = None
            else:
                goal_bay = env.random_state.choice(candidate_bays)
                bay_id = goal_bay["id"]
            
            # Reset (spawns according to the explicit geometry logic)
            obs, info = env.reset(bay_id=bay_id)
            x, y, yaw, v = env.state
            
            # Check bounds
            status = "OK"
            if abs(x) > 20.0 or abs(y) > 20.0:
                status = "WARNING: OUT OF BOUNDS"
            
            print(f"  Sample {i+1:2d}: Lot={lot_name}, Bay={env.goal_bay['id']:3s}, "
                  f"x={x:6.2f}, y={y:6.2f}, yaw={yaw:5.2f} [{status}]")
            
            # Render
            env.render()
            fig = env.fig
            ax = env.ax
            
            # Title shows which road we are on
            road_desc = "Horizontal" if forced_type == "H" else "Vertical"
            ax.set_title(f"{description}\n{lot_name} – {road_desc} road – Bay {env.goal_bay['id']}")
            
            # Save with a descriptive filename
            filename = f"{output_dir}/level_{episode_count}_sample_{i+1:02d}_{lot_name}_{forced_type}.png"
            fig.savefig(filename)
            plt.close(fig)
            
            env.close()

    print(f"\n{'='*60}")
    print("Verification complete. Check the 'spawn_verification' folder.")
    print(f"{'='*60}")

if __name__ == "__main__":
    verify_spawns()
