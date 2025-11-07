import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Our package
    pkg_auto = get_package_share_directory('autonomous_parking')

    # World file (your realistic parking lot)
    world_path = os.path.join(pkg_auto, 'worlds', 'parking_lot_a.world')

    # Launch Gazebo with this world
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={'world': world_path}.items()
    )

    # TurtleBot3 SDF model (same one used by turtlebot3_gazebo)
    tb3_sdf = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        'turtlebot3_burger',
        'model.sdf'
    )

    # Spawn robot near the entrance / in the road
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'turtlebot3',
            '-file', tb3_sdf,
            '-x', '-6.0',
            '-y', '0.0',
            '-z', '0.02'
        ],
        output='screen'
    )

    # (Robot will still publish topics from inside Gazebo; we can add
    #  a robot_state_publisher later if needed for TF.)

    return LaunchDescription([
        gazebo_launch,
        spawn_robot
    ])
