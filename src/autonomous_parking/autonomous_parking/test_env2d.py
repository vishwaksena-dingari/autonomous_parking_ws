#!/usr/bin/env python3
import random
from autonomous_parking.parking_env import ParkingEnv



def main():
    env = ParkingEnv(lot_name="lot_a")
    state = env.reset()
    print("Initial state:", state)

    for t in range(20):
        # random action: v in [-0.3, 0.3], w in [-0.5, 0.5]
        v = random.uniform(-0.3, 0.3)
        w = random.uniform(-0.5, 0.5)
        state, reward, done, info = env.step([v, w])
        print(
            f"t={t:02d}, state={['%.2f' % s for s in state]}, "
            f"reward={reward:.3f}, dist={info['dist']:.2f}, "
            f"yaw_err={info['yaw_err']:.2f}, success={info['success']}"
        )
        if done:
            print("Episode finished.")
            break


if __name__ == "__main__":
    main()

