from __future__ import annotations

from pathlib import Path
import yaml

# Try to import ROS 2 ament index (only available in a ROS environment)
try:
    from ament_index_python.packages import get_package_share_directory

    _HAS_AMENT = True
except ImportError:
    get_package_share_directory = None  # type: ignore[assignment]
    _HAS_AMENT = False


def _get_config_path() -> Path:
    """
    Find bays.yaml.

    - In a ROS 2 environment: use ament_index to find the installed share directory.
    - In a plain Python environment (e.g. Mac without ROS): use the source tree layout.

    Layouts:
      ROS install:  <share>/autonomous_parking/config/bays.yaml
      Source tree:  src/autonomous_parking/config/bays.yaml
    """
    if _HAS_AMENT:
        pkg_share = Path(get_package_share_directory("autonomous_parking"))  # type: ignore[arg-type]
        return pkg_share / "config" / "bays.yaml"

    # Fallback: running directly from the source checkout
    here = Path(__file__).resolve()
    src_root = here.parent.parent  # .../src/autonomous_parking
    return src_root / "config" / "bays.yaml"


def load_raw_bays_yaml(path: str | Path | None = None) -> dict:
    """
    Load and return the entire bays.yaml as a raw dictionary (no validation).
    Useful for debugging, introspection, or listing all available lots.
    """
    config_path = Path(path) if path is not None else _get_config_path()
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {config_path}, got {type(data).__name__}")
    return data


def load_parking_config(
    lot_name: str = "lot_a",
    path: str | Path | None = None,
) -> dict:
    """
    Load parking lot configuration from bays.yaml and return a validated dict.

    Args:
        lot_name:
            Name of the parking lot section to load (e.g. "lot_a", "lot_b").
        path:
            Optional explicit path to bays.yaml. If None, auto-detect:
              - ROS 2: via ament_index (installed package)
              - Plain Python: relative to this file in the source tree

    Returns:
        dict: {
            "entrance": {"x": float, "y": float, "yaw": float},
            "bays": [{"id": str, "x": float, "y": float, "yaw": float}, ...]
        }

    Raises:
        FileNotFoundError: If bays.yaml doesn't exist.
        KeyError: If lot_name not found or required keys missing.
        ValueError: If YAML structure is invalid.
    """
    config_path = Path(path) if path is not None else _get_config_path()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected dict at top level of {config_path}, "
            f"got {type(data).__name__}"
        )

    if lot_name not in data:
        available = list(data.keys())
        raise KeyError(
            f"Lot '{lot_name}' not found in {config_path}. "
            f"Available lots: {available}"
        )

    lot_cfg = data[lot_name]

    required_keys = {"entrance", "bays"}
    missing_keys = required_keys - set(lot_cfg.keys())
    if missing_keys:
        raise KeyError(f"Lot '{lot_name}' missing required keys: {missing_keys}")

    entrance = lot_cfg["entrance"]
    if not isinstance(entrance, dict) or not all(
        k in entrance for k in ("x", "y", "yaw")
    ):
        raise ValueError(
            f"Lot '{lot_name}': 'entrance' must be a dict with "
            f"'x', 'y', 'yaw' keys. Got: {entrance!r}"
        )

    bays = lot_cfg["bays"]
    if not isinstance(bays, list):
        raise ValueError(
            f"Lot '{lot_name}': 'bays' must be a list, " f"got {type(bays).__name__}"
        )
    if not bays:
        raise ValueError(f"No bays defined for lot '{lot_name}'")

    for i, bay in enumerate(bays):
        if not isinstance(bay, dict):
            raise ValueError(
                f"Lot '{lot_name}': bay {i} must be a dict, "
                f"got {type(bay).__name__}"
            )
        if not all(k in bay for k in ("id", "x", "y", "yaw")):
            raise ValueError(
                f"Lot '{lot_name}': bay {i} must contain 'id', 'x', 'y', 'yaw'. "
                f"Got keys: {list(bay.keys())}"
            )

    return lot_cfg
