import os
import yaml
from ament_index_python.packages import get_package_share_directory


def load_parking_config(lot_name: str = "lot_a"):
    """
    Load parking lot configuration from config/bays.yaml.
    
    Args:
        lot_name: Name of the parking lot ('lot_a' or 'lot_b')
        
    Returns:
        dict: {
            "entrance": {"x": float, "y": float, "yaw": float},
            "bays": [{"id": str, "x": float, "y": float, "yaw": float}, ...]
        }
        
    Raises:
        FileNotFoundError: If bays.yaml doesn't exist
        KeyError: If lot_name not found or required keys missing
        ValueError: If YAML structure is invalid
    """
    # Get config path
    pkg_share = get_package_share_directory('autonomous_parking')
    config_path = os.path.join(pkg_share, 'config', 'bays.yaml')
    
    # Check file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict in {config_path}, got {type(data).__name__}"
        )
    
    # Check lot exists
    if lot_name not in data:
        available = list(data.keys())
        raise KeyError(
            f"Lot '{lot_name}' not found. Available lots: {available}"
        )
    
    lot_cfg = data[lot_name]
    
    # Validate required keys
    required_keys = {'entrance', 'bays'}
    missing_keys = required_keys - set(lot_cfg.keys())
    if missing_keys:
        raise KeyError(
            f"Lot '{lot_name}' missing required keys: {missing_keys}"
        )
    
    # Validate entrance structure
    entrance = lot_cfg['entrance']
    if not all(k in entrance for k in ('x', 'y', 'yaw')):
        raise ValueError(
            f"Entrance must contain 'x', 'y', 'yaw'. Got: {list(entrance.keys())}"
        )
    
    # Validate bays structure
    bays = lot_cfg['bays']
    if not isinstance(bays, list):
        raise ValueError(f"'bays' must be a list, got {type(bays).__name__}")
    
    if not bays:
        raise ValueError(f"No bays defined for lot '{lot_name}'")
    
    for i, bay in enumerate(bays):
        if not all(k in bay for k in ('id', 'x', 'y', 'yaw')):
            raise ValueError(
                f"Bay {i} must contain 'id', 'x', 'y', 'yaw'. Got: {list(bay.keys())}"
            )
    
    return lot_cfg

# import os
# import yaml
# from ament_index_python.packages import get_package_share_directory


# def load_parking_config(lot_name: str = "lot_a"):
#     """
#     Load parking lot configuration from config/bays.yaml.

#     Returns:
#       {
#         "entrance": {...},
#         "bays": [ {...}, ... ]
#       }
#     """
#     pkg_share = get_package_share_directory('autonomous_parking')
#     config_path = os.path.join(pkg_share, 'config', 'bays.yaml')

#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")

#     with open(config_path, 'r') as f:
#         data = yaml.safe_load(f)

#     # Expecting top-level key "lot_a", "lot_b", etc.
#     if not isinstance(data, dict):
#         raise ValueError(f"Unexpected YAML structure in {config_path}: {type(data)}")

#     if lot_name not in data:
#         raise KeyError(
#             f"Lot '{lot_name}' not found in {config_path}. "
#             f"Available keys: {list(data.keys())}"
#         )

#     lot_cfg = data[lot_name]

#     # Basic sanity
#     if "entrance" not in lot_cfg or "bays" not in lot_cfg:
#         raise KeyError(
#             f"Lot '{lot_name}' must contain 'entrance' and 'bays' keys, "
#             f"got: {list(lot_cfg.keys())}"
#         )

#     return lot_cfg
