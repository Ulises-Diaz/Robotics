from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='astar',
            executable='simple_map',
            name='simple_map_publisher',
            output='screen'
        ),
        Node(
            package='astar',
            executable='astar',
            name='astar_planner',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )
    ])
