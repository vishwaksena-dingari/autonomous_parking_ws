import numpy as np
import matplotlib.pyplot as plt
from autonomous_parking.env2d.parking_env import ParkingEnv
import os

def verify_spawns():
    print("🚀 Starting Spawn Verification Stress Test (Direct Physics Check)...")
    
    scenarios = [
        {"name": "Lot A (All Bays)", "lot": "lot_a", "bay": None},
        {"name": "Lot B (H-Bays)", "lot": "lot_b", "bay": "H3"}, # Middle H-bay
        {"name": "Lot B (V-Bays)", "lot": "lot_b", "bay": "V3"}, # Middle V-bay
    ]

    for sc in scenarios:
        name = sc["name"]
        lot = sc["lot"]
        bay = sc["bay"]
        
        print(f"\nTesting {name}...")
        env = ParkingEnv(lot_name=lot, render_mode=None)
        
        # Force the curriculum spawn logic (which has the strict clamps)
        env.max_spawn_dist_override = 20.0 
        
        x_positions = []
        y_positions = []
        
        # Run 5000 resets (Massive Stress Test)
        for i in range(5000):
            # Pass bay_id to force specific geometry testing
            env.reset(options={"bay_id": bay} if bay else None)
            
            state = env.state
            x, y = state[0], state[1]
            x_positions.append(x)
            y_positions.append(y)
            
            if i % 1000 == 0:
                print(f"  Sample {i}/5000: ({x:.2f}, {y:.2f})")

        # Convert to numpy
        xs = np.array(x_positions)
        ys = np.array(y_positions)
        
        # --- STATISTICS ---
        print(f"  [STATS] X Range: [{xs.min():.2f}, {xs.max():.2f}]")
        print(f"  [STATS] Y Range: [{ys.min():.2f}, {ys.max():.2f}]")
        
        # Check against limits
        success = True
        if lot == "lot_a":
            # Limit: Y in [-2.0, 2.0]
            if ys.min() < -2.01 or ys.max() > 2.01:
                print(f"  ❌ FAILURE: Y out of bounds! Found [{ys.min():.2f}, {ys.max():.2f}]")
                success = False
            else:
                print("  ✅ SUCCESS: Spawns strictly within +/- 2.0m Y.")
                
        elif lot == "lot_b":
            if bay and bay.startswith("H"):
                # H-bays: Y in [8.0, 12.0]
                if ys.min() < 7.99 or ys.max() > 12.01:
                    print(f"  ❌ FAILURE: Y out of bounds (H-Bay)! Found [{ys.min():.2f}, {ys.max():.2f}]")
                    success = False
                else:
                    print("  ✅ SUCCESS: H-Bay spawns strictly within [8.0, 12.0] Y.")
            elif bay and bay.startswith("V"):
                # V-bays: X in [-2.0, 2.0], Y <= 11.0 (Updated)
                if xs.min() < -2.01 or xs.max() > 2.01:
                    print(f"  ❌ FAILURE: X out of bounds (V-Bay)! Found [{xs.min():.2f}, {xs.max():.2f}]")
                    success = False
                if ys.max() > 11.01:
                    print(f"  ❌ FAILURE: Y overshoot (V-Bay)! Found max Y={ys.max():.2f}")
                    success = False
                if success:
                    print("  ✅ SUCCESS: V-Bay spawns strictly within road bounds.")

        # --- PLOTTING ---
        plt.figure(figsize=(10, 10))
        plt.title(f"Spawn Distribution - {name} (5000 samples)")
        
        # Draw Road Limits (Visual Reference)
        if lot == "lot_a":
            plt.axhspan(-3.75, 3.75, color='gray', alpha=0.3, label='Road')
            plt.axhspan(-2.0, 2.0, color='green', alpha=0.1, label='Safe Zone')
        elif lot == "lot_b":
            # H-Road
            plt.axhspan(6.25, 13.75, color='gray', alpha=0.3)
            plt.axhspan(8.0, 12.0, color='green', alpha=0.1)
            # V-Road
            plt.axvspan(-3.75, 3.75, ymax=0.8, color='gray', alpha=0.3)
            plt.axvspan(-2.0, 2.0, ymax=0.75, color='green', alpha=0.1)

        plt.scatter(xs, ys, alpha=0.5, s=10, c='blue', label='Spawns')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.grid(True)
        plt.legend()
        
        filename = f"spawn_verify_{lot}_{bay if bay else 'all'}.png"
        plt.savefig(filename)
        print(f"  📸 Saved plot to {filename}")
        plt.close()
        
        env.close()

if __name__ == "__main__":
    verify_spawns()
