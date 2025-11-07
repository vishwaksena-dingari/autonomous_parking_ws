from setuptools import setup
import os
from glob import glob

package_name = "autonomous_parking"

setup(
    name=package_name,
    version="0.0.0",
    # IMPORTANT: include the env2d subpackage so it gets installed
    packages=[
        package_name,
        f"{package_name}.env2d",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.world")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "pyyaml",
        "numpy",
        "matplotlib",
    ],
    zip_safe=True,
    maintainer="vd",
    maintainer_email="you@example.com",  # change if you want
    description="Autonomous parking project with 2D and Gazebo environments",
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
