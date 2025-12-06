
import numpy as np
import matplotlib.pyplot as plt
from autonomous_parking.env2d.waypoint_env import WaypointEnv

def verify_v40_physics():
    print("\n🔹 Verifying v40 Physics & Rewards...")
    
    # 1. Initialize Env
    env = WaypointEnv(render_mode="rgb_array", verbose=False)
    env.reset(seed=42)
    
    # 2. Setup Scenario: Occupy Bay 1, Goal is Bay 5 (random, but let's force collision)
    # We will FORCE the agent to spawn ON TOP of an occupied bay.
    
    # Get a bay to occupy
    target_bay = env.bays[0] 
    env.occupied_manager.set_specific_occupancy([target_bay['id']])
    env.occupied_bays = env.occupied_manager.get_all_parked_cars()
    
    print(f"   Target Occupied Bay: {target_bay['id']} at ({target_bay['x']:.2f}, {target_bay['y']:.2f})")
    
    # 3. Teleport Agent INSIDE the occupied bay
    # Only WaypointEnv logic might override state, but let's set it directly.
    # v40 FIX: Target bay yaw is 0 (Horizontal), but geometry is Vertical. 
    # Must add 90 deg (pi/2) to align car with bay.
    env.state = np.array([target_bay['x'], target_bay['y'], target_bay['yaw'] + np.pi/2, 0.0])
    
    # 4. Step
    print("   Step 0: Teleported inside obstacle. Stepping...")
    # v40 FIX: Action is [steering_norm, accel_norm]. 
    # To move forward: steer=0.0, accel=0.5. 
    # With max_speed=5, this is v_cmd=2.5 m/s.
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.5]))
    
    # 5. Assertions
    print(f"   Result: Terminated={terminated}, Reward={reward:.2f}")
    print(f"   Info: {info}")
    
    # Check physics: Velocity should be > 0 in returning state
    # WaypointEnv state: x, y, yaw, v
    v_actual = env.state[3]
    print(f"   Actual Velocity: {v_actual:.2f} m/s (Expected > 0 because accel=0.5)")
    
    if v_actual < 0.1:
        print("❌ FAILED: Car did not move/accelerate! Action mapping might be wrong.")
    else:
        print("✅ PASSED: Car accelerated correctly.")

    if not terminated:
        print("❌ FAILED: Agent did not terminate upon collision!")
    else:
        print("✅ PASSED: Agent terminated immediately.")
        
    if not info.get("crash", False):
        print("❌ FAILED: 'crash' flag missing in info!")
    else:
        print("✅ PASSED: 'crash' flag detected.")
        
    # Check Lidar Blindness
    # Extract lidar from WaypointEnv obs (98D)
    # Lidar starts at index 34 in the 98D vector
    waypoint_obs = obs
    lidar_slice = waypoint_obs[34:] 
    min_reading = np.min(lidar_slice)
    print(f"   Lidar Min Reading (from obs): {min_reading:.2f}")
    
    if min_reading > 5.0:
       # If we are INSIDE a car, lidar should be tiny/zero. 
       # If it reads 20.0 (max), we are blind.
       print("❌ FAILED: Lidar shows empty space despite being inside car! Indexing bug?")
    else:
       print("✅ PASSED: Lidar sees obstacle (reading < 5.0).")

    if reward > -400: # Assuming -500 penalty + small step rewards
        print(f"❌ FAILED: Reward ({reward}) too high! Expected ~ -500.")
    else:
        print("✅ PASSED: Harsh collision penalty applied.")

    # 6. Render Check
    try:
        env.render()
        # Save frame
        if env.fig:
            env.fig.savefig("v40_verification_pass.png")
            print("✅ PASSED: Render successful. Saved to v40_verification_pass.png")
        else:
            print("⚠️ WARNING: No figure to save.")
    except Exception as e:
        print(f"❌ FAILED: Render crashed: {e}")

    env.close()

if __name__ == "__main__":
    verify_v40_physics()
