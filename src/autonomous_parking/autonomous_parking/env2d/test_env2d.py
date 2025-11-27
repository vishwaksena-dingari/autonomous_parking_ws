#!/usr/bin/env python3
"""
Test script for ParkingEnv - validates environment functionality
"""

import time
import math
import argparse

from autonomous_parking.env2d.parking_env import ParkingEnv


def test_simple_motion(env, num_steps=50):
    """Test basic forward motion with slight steering."""
    print("\n=== Test: Simple Motion ===")
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    print(f"Goal bay: {env.goal_bay['id']}")
    
    for t in range(num_steps):
        # Simple forward motion with slight right turn
        v_cmd = 1.0
        steer_cmd = math.radians(5.0)
        
        obs, reward, terminated, truncated, info = env.step((v_cmd, steer_cmd))
        done = terminated or truncated
        
        print(
            f"t={t:03d} | "
            f"dist={info['dist']:5.2f}m | "
            f"yaw_err={math.degrees(info['yaw_err']):6.1f}° | "
            f"reward={reward:6.2f} | "
            f"success={info['success']}"
        )
        
        env.render()
        time.sleep(0.05)
        
        if done:
            if info['success']:
                print(f"✓ SUCCESS in {t+1} steps!")
            else:
                print(f"✗ Episode ended without success (timeout or OOB)")
            break
    
    return info.get('success', False)


def test_specific_bay(env, bay_id, num_steps=100):
    """Test parking in a specific bay."""
    print(f"\n=== Test: Parking in Bay {bay_id} ===")
    
    try:
        obs, _ = env.reset(bay_id=bay_id)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    print(f"Target bay: {env.goal_bay}")
    print(f"Initial obs: {obs}")
    
    for t in range(num_steps):
        # Simple proportional controller (demo only)
        local_x, local_y, yaw_err, v, dist = obs
        
        # Steer towards goal
        steer_cmd = 0.5 * math.atan2(local_y, local_x)
        
        # Slow down when close
        if dist < 2.0:
            v_cmd = 0.3
        else:
            v_cmd = 1.0
        
        obs, reward, terminated, truncated, info = env.step((v_cmd, steer_cmd))
        done = terminated or truncated
        
        if t % 10 == 0:  # Print every 10 steps
            print(
                f"t={t:03d} | "
                f"dist={info['dist']:5.2f}m | "
                f"yaw_err={math.degrees(info['yaw_err']):6.1f}° | "
                f"reward={reward:6.2f}"
            )
        
        env.render()
        time.sleep(0.05)
        
        if done:
            if info['success']:
                print(f"✓ SUCCESS in {t+1} steps!")
            else:
                print(f"✗ Failed to park successfully")
            break
    
    return info.get('success', False)


def test_multiple_episodes(env, num_episodes=3):
    """Test multiple random episodes."""
    print(f"\n=== Test: {num_episodes} Random Episodes ===")
    
    success_count = 0
    
    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        obs = env.reset()  # Random bay
        print(f"Target: {env.goal_bay['id']}")
        
        for t in range(100):
            # Random actions for testing
            v_cmd = 0.5
            steer_cmd = math.radians(10.0) if t % 20 < 10 else math.radians(-10.0)
            
            obs, reward, done, info = env.step((v_cmd, steer_cmd))
            
            env.render()
            time.sleep(0.02)  # Faster for multiple episodes
            
            if done:
                if info['success']:
                    print(f"✓ Episode {ep+1} SUCCESS!")
                    success_count += 1
                else:
                    print(f"✗ Episode {ep+1} failed")
                break
    
    print(f"\n=== Results: {success_count}/{num_episodes} successful ===")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Test ParkingEnv")
    parser.add_argument(
        "--lot",
        type=str,
        default="lot_a",
        choices=["lot_a", "lot_b"],
        help="Parking lot to test",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="simple",
        choices=["simple", "bay", "multiple"],
        help="Test type to run",
    )
    parser.add_argument(
        "--bay-id",
        type=str,
        default="A1",
        help="Specific bay ID for 'bay' test",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of steps per episode",
    )
    
    args = parser.parse_args()
    
    print(f"Initializing ParkingEnv with lot={args.lot}")
    env = ParkingEnv(lot_name=args.lot, dt=0.1)
    
    try:
        if args.test == "simple":
            test_simple_motion(env, num_steps=args.steps)
        
        elif args.test == "bay":
            test_specific_bay(env, bay_id=args.bay_id, num_steps=args.steps)
        
        elif args.test == "multiple":
            test_multiple_episodes(env, num_episodes=3)
        
        print("\n✓ All tests completed")
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        raise
    
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()


# import time
# import math

# from autonomous_parking.env2d.parking_env import ParkingEnv


# def main():
#     env = ParkingEnv(lot_name="lot_a", dt=0.1)
#     obs = env.reset()
#     print("Initial obs:", obs)

#     for t in range(50):
#         # simple forward / slight steering to see motion
#         v_cmd = 1.0
#         steer_cmd = math.radians(5.0)

#         obs, reward, done, info = env.step((v_cmd, steer_cmd))
#         print(
#             f"t={t:03d}, "
#             f"obs={obs}, "
#             f"reward={reward:.3f}, "
#             f"done={done}, "
#             f"success={info['success']}"
#         )
#         env.render()
#         time.sleep(0.05)

#         if done:
#             print("Episode finished, resetting...")
#             obs = env.reset()

#     env.close()
