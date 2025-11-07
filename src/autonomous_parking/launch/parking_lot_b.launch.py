import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_auto = get_package_share_directory('autonomous_parking')

    # Use the new L-shaped world
    world_path = os.path.join(pkg_auto, 'worlds', 'parking_lot_b.world')

    # Start Gazebo with that world
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

    # TurtleBot3 SDF model (from turtlebot3_gazebo)
    tb3_model = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models',
        'turtlebot3_burger',
        'model.sdf'
    )

    # Spawn TB3 near the entrance on the vertical road
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-file', tb3_model,
            '-entity', 'turtlebot3',
            '-x', '0.0',
            '-y', '-18.0',
            '-z', '0.0',
            '-Y', '1.5708'   # face north along the road
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_robot,
    ])
