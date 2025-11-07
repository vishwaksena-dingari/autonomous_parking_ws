#!/usr/bin/env python3
"""
Keyboard teleoperation for ParkingEnv

Controls:
  ↑ : accelerate forward
  ↓ : accelerate backward
  ← : steer left
  → : steer right
  SPACE : brake (v = 0)
  r : reset episode (new random bay)
  b : reset to specific bay (cycles through bays)
  q : quit
"""

import math
import argparse

import numpy as np
import matplotlib.pyplot as plt

from autonomous_parking.env2d.parking_env import ParkingEnv


class KeyboardController:
    """
    Keyboard teleoperation for 2D parking environment.
    
    Attributes:
        env: ParkingEnv instance
        v: Current commanded velocity (m/s)
        delta: Current commanded steering angle (rad)
    """
    
    def __init__(self, env: ParkingEnv):
        self.env = env
        self.v = 0.0        # current commanded speed
        self.delta = 0.0    # current commanded steering angle
        
        # Bay cycling for 'b' key
        self.bay_index = 0
        
        # Acceleration parameters
        self.v_increment = 0.3      # m/s per keypress
        self.delta_increment = math.radians(5.0)  # rad per keypress
        
        # Stats
        self.total_steps = 0
        self.successful_parks = 0
        self.total_episodes = 0
    
    def on_key(self, event):
        """Handle keyboard events."""
        
        if event.key == "up":
            self.v += self.v_increment
            
        elif event.key == "down":
            self.v -= self.v_increment
            
        elif event.key == "left":
            self.delta += self.delta_increment
            
        elif event.key == "right":
            self.delta -= self.delta_increment
            
        elif event.key == " ":
            # Brake
            self.v = 0.0
            self.delta = 0.0
            print("[brake] v=0, delta=0")
            return
            
        elif event.key == "r":
            # Reset to random bay
            print("[reset] New random goal bay")
            self.total_episodes += 1
            self.v = 0.0
            self.delta = 0.0
            obs = self.env.reset()
            self.env.render()
            print(f"  → Target bay: {self.env.goal_bay['id']}")
            print(f"  → Episodes: {self.total_episodes}, Success: {self.successful_parks}")
            return
            
        elif event.key == "b":
            # Reset to specific bay (cycle through)
            bays = self.env.bays
            bay = bays[self.bay_index % len(bays)]
            self.bay_index += 1
            
            print(f"[reset] Target bay: {bay['id']}")
            self.total_episodes += 1
            self.v = 0.0
            self.delta = 0.0
            obs = self.env.reset(bay_id=bay['id'])
            self.env.render()
            return
            
        elif event.key == "q":
            # Quit
            print("\n=== Session Summary ===")
            print(f"Total episodes: {self.total_episodes}")
            print(f"Successful parks: {self.successful_parks}")
            print(f"Total steps: {self.total_steps}")
            if self.total_episodes > 0:
                print(f"Success rate: {100*self.successful_parks/self.total_episodes:.1f}%")
            print("[quit]")
            plt.close(self.env.fig)
            return
            
        elif event.key == "h":
            # Help
            self._print_help()
            return
            
        else:
            # Ignore other keys
            return
        
        # Clip commands to physical limits
        self.v = float(np.clip(self.v, -self.env.max_speed, self.env.max_speed))
        self.delta = float(np.clip(self.delta, -self.env.max_steer, self.env.max_steer))
        
        # Take multiple small steps per keypress for smoother control
        steps_per_press = 5
        for _ in range(steps_per_press):
            obs, reward, done, info = self.env.step((self.v, self.delta))
            self.env.render()
            self.total_steps += 1
            
            # Print state
            x, y, yaw, v_actual = self.env.state
            local_x, local_y, yaw_err, v_obs, dist = obs
            
            print(
                f"v={self.v:+5.2f} m/s | "
                f"δ={math.degrees(self.delta):+6.1f}° | "
                f"pos=({x:6.2f},{y:6.2f}) | "
                f"dist={dist:5.2f}m | "
                f"yaw_err={math.degrees(yaw_err):+6.1f}° | "
                f"reward={reward:+7.2f}"
            )
            
            if done:
                if info['success']:
                    print(f"✓ SUCCESS! Parked in {info['steps']} steps")
                    self.successful_parks += 1
                else:
                    print(f"✗ Episode ended (timeout or out-of-bounds)")
                
                print("[auto-reset]")
                self.total_episodes += 1
                self.v = 0.0
                self.delta = 0.0
                self.env.reset()
                self.env.render()
                break
    
    def _print_help(self):
        """Print control help."""
        print("""
╔═══════════════════════════════════════════════════════════╗
║                 KEYBOARD CONTROLS                         ║
╠═══════════════════════════════════════════════════════════╣
║  ↑         : Accelerate forward                           ║
║  ↓         : Accelerate backward (reverse)                ║
║  ←         : Steer left                                   ║
║  →         : Steer right                                  ║
║  SPACE     : Brake (stop all motion)                      ║
║  r         : Reset episode (random bay)                   ║
║  b         : Reset to specific bay (cycles)               ║
║  h         : Show this help                               ║
║  q         : Quit                                         ║
╚═══════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for parking environment"
    )
    parser.add_argument(
        "--lot",
        type=str,
        default="lot_a",
        choices=["lot_a", "lot_b"],
        help="Parking lot to use",
    )
    parser.add_argument(
        "--bay",
        type=str,
        default=None,
        help="Start with specific bay (e.g., 'A1')",
    )
    
    args = parser.parse_args()
    
    print(f"Initializing ParkingEnv with lot={args.lot}")
    env = ParkingEnv(lot_name=args.lot, dt=0.1)
    
    # Initial reset
    if args.bay:
        try:
            env.reset(bay_id=args.bay)
            print(f"Starting with target bay: {args.bay}")
        except ValueError as e:
            print(f"Warning: {e}")
            print("Resetting to random bay instead")
            env.reset()
    else:
        env.reset()
    
    env.render()
    
    # Setup controller
    ctrl = KeyboardController(env)
    
    # Print instructions
    ctrl._print_help()
    print(f"Target bay: {env.goal_bay['id']}")
    print("\n→ Focus the matplotlib window and use keyboard controls")
    print("→ Press 'h' for help\n")
    
    # Connect keyboard handler
    env.fig.canvas.mpl_connect("key_press_event", ctrl.on_key)
    
    # Show plot (blocking)
    try:
        plt.show(block=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()