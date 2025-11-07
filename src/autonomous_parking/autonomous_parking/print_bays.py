#!/usr/bin/env python3
"""
Inspect parking lot configuration from bays.yaml
"""
from autonomous_parking.config_loader import load_parking_config


def main():
    """Load and display parking lot configuration."""
    lot_name = "lot_a"  # Change to "lot_b" to inspect second layout
    
    try:
        cfg = load_parking_config(lot_name)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error loading config: {e}")
        return
    
    print(f"=== Parking Lot: {lot_name} ===\n")
    
    # Display entrance
    entrance = cfg["entrance"]
    print("Entrance:")
    print(f"  Position: ({entrance['x']:.2f}, {entrance['y']:.2f})")
    print(f"  Yaw: {entrance['yaw']:.4f} rad ({entrance['yaw'] * 57.2958:.1f}°)")
    
    # Display bays
    bays = cfg["bays"]
    print(f"\nBays ({len(bays)} total):")
    for bay in bays:
        print(f"  {bay['id']:3s}: pos=({bay['x']:6.2f}, {bay['y']:6.2f}), "
              f"yaw={bay['yaw']:.4f} rad ({bay['yaw'] * 57.2958:6.1f}°)")
    
    print(f"\n✓ Configuration loaded successfully")


if __name__ == "__main__":
    main()

# from autonomous_parking.config_loader import load_parking_config


# def main():
#     lot_name = "lot_a"
#     cfg = load_parking_config(lot_name)

#     entrance = cfg.get("entrance", {})
#     bays = cfg.get("bays", [])

#     print(f"=== Parking Lot: {lot_name} ===\n")

#     print("Entrance:")
#     print(entrance)

#     print("\nBays:")
#     for bay in bays:
#         print(bay)


# if __name__ == "__main__":
#     main()
