from setuptools import setup
import os
from glob import glob

package_name = "autonomous_parking"

setup(
    name=package_name,
    version="0.0.0",
    packages=[
        package_name,
        f"{package_name}.env2d",
    ],
    data_files=[
        # ament index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package.xml
        ("share/" + package_name, ["package.xml"]),
        # ALL launch files (parking_lot_a + parking_lot_b, etc.)
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        # ALL world files
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        # ALL yaml configs
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vd",
    maintainer_email="vd@todo.todo",
    description="Autonomous parking project",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "print_bays = autonomous_parking.print_bays:main",
            "test_env2d = autonomous_parking.test_env2d:main",
            "keyboard_drive_2d = autonomous_parking.keyboard_drive_2d:main",
        ],
    },
)
