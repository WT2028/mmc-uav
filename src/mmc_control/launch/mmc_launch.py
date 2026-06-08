import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Resolve resources from installed ROS 2 package shares so the open-source
    # skeleton is not tied to the original development workspace path.
    control_share_dir = get_package_share_directory('mmc_control')
    description_share_dir = get_package_share_directory('mmc_uav_description')

    urdf_path = os.path.join(description_share_dir, 'urdf', 'mmc_uav.urdf')
    default_world_path = os.path.join(description_share_dir, 'worlds', 'empty_world.sdf')
    bridge_yaml_path = os.path.join(control_share_dir, 'config', 'ros_gz_bridge_mmc.yaml')
    rviz_config_path = os.path.join(control_share_dir, 'config', 'mmc_rviz.rviz')
    pj_layout_path = os.path.join(control_share_dir, 'config', 'mmc_pj_layout.xml')
    gazebo_resource_paths = [os.path.dirname(description_share_dir)]
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH')
    if existing_resource_path:
        gazebo_resource_paths.append(existing_resource_path)
    gazebo_resource_path = os.pathsep.join(gazebo_resource_paths)
    uav_model_path = LaunchConfiguration('uav_model_path')
    world_path = LaunchConfiguration('world_sdf_path')
    enable_gui = LaunchConfiguration('enable_gui')
    enable_rviz = LaunchConfiguration('enable_rviz')
    enable_plotjuggler = LaunchConfiguration('enable_plotjuggler')
    enable_wind_bridge = LaunchConfiguration('enable_wind_bridge')
    ndo_enabled = LaunchConfiguration('ndo_enabled')
    ndo_compensation_limit_deg = LaunchConfiguration('ndo_compensation_limit_deg')
    ndo_compensation_limit_schedule_enabled = LaunchConfiguration('ndo_compensation_limit_schedule_enabled')
    ndo_compensation_limit_low_speed = LaunchConfiguration('ndo_compensation_limit_low_speed')
    ndo_compensation_limit_high_speed = LaunchConfiguration('ndo_compensation_limit_high_speed')
    ndo_compensation_limit_low_deg = LaunchConfiguration('ndo_compensation_limit_low_deg')
    ndo_compensation_limit_high_deg = LaunchConfiguration('ndo_compensation_limit_high_deg')
    ndo_compensated_attitude_limit_deg = LaunchConfiguration('ndo_compensated_attitude_limit_deg')
    ndo_feedback_relief_enabled = LaunchConfiguration('ndo_feedback_relief_enabled')
    ndo_feedback_relief_gain = LaunchConfiguration('ndo_feedback_relief_gain')
    ndo_feedback_relief_deadband_deg = LaunchConfiguration('ndo_feedback_relief_deadband_deg')
    ndo_feedback_relief_max_fraction = LaunchConfiguration('ndo_feedback_relief_max_fraction')
    ndo_transient_attitude_boost_enabled = LaunchConfiguration('ndo_transient_attitude_boost_enabled')
    ndo_transient_attitude_limit_deg = LaunchConfiguration('ndo_transient_attitude_limit_deg')
    ndo_transient_attitude_boost_duration = LaunchConfiguration('ndo_transient_attitude_boost_duration')
    ndo_transient_attitude_boost_fade = LaunchConfiguration('ndo_transient_attitude_boost_fade')
    ndo_base_gain_force = LaunchConfiguration('ndo_base_gain_force')
    ndo_base_gain_torque = LaunchConfiguration('ndo_base_gain_torque')
    ndo_lpf_alpha = LaunchConfiguration('ndo_lpf_alpha')
    ndo_attitude_gain = LaunchConfiguration('ndo_attitude_gain')
    ndo_adaptive_gain_enabled = LaunchConfiguration('ndo_adaptive_gain_enabled')
    ndo_coupling_compensation_enabled = LaunchConfiguration('ndo_coupling_compensation_enabled')
    inner_thrust_retarget_enabled = LaunchConfiguration('inner_thrust_retarget_enabled')
    slider_command_shaping_enabled = LaunchConfiguration('slider_command_shaping_enabled')
    slider_command_shape_tau = LaunchConfiguration('slider_command_shape_tau')
    slider_command_rate_limit = LaunchConfiguration('slider_command_rate_limit')
    outer_xy_kp = LaunchConfiguration('outer_xy_kp')
    outer_xy_ki = LaunchConfiguration('outer_xy_ki')
    outer_xy_kd = LaunchConfiguration('outer_xy_kd')
    outer_terminal_hover_velocity_damping = LaunchConfiguration('outer_terminal_hover_velocity_damping')
    body_mass_kg = LaunchConfiguration('body_mass_kg')
    moving_mass_kg = LaunchConfiguration('moving_mass_kg')
    slider_wn_mass = LaunchConfiguration('slider_wn_mass')
    slider_zeta_mass = LaunchConfiguration('slider_zeta_mass')
    rotor_max_speed_rad_s = LaunchConfiguration('rotor_max_speed_rad_s')
    rotor_motor_time_constant_up = LaunchConfiguration('rotor_motor_time_constant_up')
    rotor_motor_time_constant_down = LaunchConfiguration('rotor_motor_time_constant_down')
    rotor_motor_rate_limit_rad_s2 = LaunchConfiguration('rotor_motor_rate_limit_rad_s2')
    auto_scene_mode = LaunchConfiguration('auto_scene_mode')
    auto_scene_hover_hold_time = LaunchConfiguration('auto_scene_hover_hold_time')
    auto_scene_move_duration = LaunchConfiguration('auto_scene_move_duration')
    auto_scene_horizontal_accel_limit = LaunchConfiguration('auto_scene_horizontal_accel_limit')
    auto_scene_target_x = LaunchConfiguration('auto_scene_target_x')
    auto_scene_target_y = LaunchConfiguration('auto_scene_target_y')
    auto_scene_target_z = LaunchConfiguration('auto_scene_target_z')
    auto_scene_yaw_ref_mode = LaunchConfiguration('auto_scene_yaw_ref_mode')
    auto_scene_yaw_step_deg = LaunchConfiguration('auto_scene_yaw_step_deg')
    auto_scene_yaw_ramp_duration = LaunchConfiguration('auto_scene_yaw_ramp_duration')
    # 统一用一个开关同时控制：
    # 1) controller 是否接收手动 XY 指令
    # 2) keyboard teleop 窗口是否随 launch 自动启动
    manual_xy_enabled = False

    # 1. 设置模型资源路径环境变量
    set_env = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=gazebo_resource_path)

    # 2. 显式分离 Gazebo server / GUI：
    #    - `gz sim <world>` 会额外 fork `gz sim server`，launch 只跟踪前台包装进程，
    #      在 timeout / 中断场景下容易留下后台 server。
    #    - 改成 `-s` 前台 server + `-g` 独立 GUI 后，launch 能正确托管生命周期，
    #      下次重启时不会复用旧世界里的无人机。
    start_gazebo_server = ExecuteProcess(
        cmd=['gz', 'sim', '-s', world_path, '-r'],
        output='screen'
    )

    start_gazebo_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-g'],
        output='screen',
        condition=IfCondition(enable_gui),
    )

    # 3. 将无人机模型 Spawn 到世界中（延时 3 秒，等待 Gazebo 完成加载）
    spawn_model = ExecuteProcess(
        cmd=[
            'gz', 'service', '-s', '/world/mmc_world/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', ['sdf_filename: "', uav_model_path, '", pose: {position: {z: 1.0}}, name: "mmc_uav"']
        ],
        output='screen'
    )

    # 4. 启动 ros_gz_bridge（延时 4 秒）
    # 注意：前提是在终端中已正确 source 工作空间
    start_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_yaml_path}'],
        output='screen'
    )

    # 默认不启动风场桥接；M4 扰动/NDO 对比通过 launch 参数显式启用。
    start_wind_bridge = Node(
        package='mmc_control',
        executable='wind_bridge_node',
        output='screen',
        condition=IfCondition(enable_wind_bridge),
    )

    # 5. 启动 RViz2 可视化
    start_rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen',
        condition=IfCondition(enable_rviz),
    )

    # 6. 启动 PlotJuggler 可视化
    start_plotjuggler = Node(
        package='plotjuggler',
        executable='plotjuggler',
        arguments=['-l', pj_layout_path],
        output='screen',
        condition=IfCondition(enable_plotjuggler),
    )

    # 7. 启动 Python 飞控控制节点（延时 8 秒，等待传感器数据稳定）
    start_controller = Node(
        package='mmc_control',
        executable='controller_node',
        parameters=[  {
            'use_sim_time': True,
            'world_sdf_path': ParameterValue(world_path, value_type=str),
            'manual_xy_enabled': manual_xy_enabled,
            'ndo_enabled': ParameterValue(ndo_enabled, value_type=bool),
            'ndo_compensation_limit_deg': ParameterValue(ndo_compensation_limit_deg, value_type=float),
            'ndo_compensation_limit_schedule_enabled': ParameterValue(ndo_compensation_limit_schedule_enabled, value_type=bool),
            'ndo_compensation_limit_low_speed': ParameterValue(ndo_compensation_limit_low_speed, value_type=float),
            'ndo_compensation_limit_high_speed': ParameterValue(ndo_compensation_limit_high_speed, value_type=float),
            'ndo_compensation_limit_low_deg': ParameterValue(ndo_compensation_limit_low_deg, value_type=float),
            'ndo_compensation_limit_high_deg': ParameterValue(ndo_compensation_limit_high_deg, value_type=float),
            'ndo_compensated_attitude_limit_deg': ParameterValue(ndo_compensated_attitude_limit_deg, value_type=float),
            'ndo_feedback_relief_enabled': ParameterValue(ndo_feedback_relief_enabled, value_type=bool),
            'ndo_feedback_relief_gain': ParameterValue(ndo_feedback_relief_gain, value_type=float),
            'ndo_feedback_relief_deadband_deg': ParameterValue(ndo_feedback_relief_deadband_deg, value_type=float),
            'ndo_feedback_relief_max_fraction': ParameterValue(ndo_feedback_relief_max_fraction, value_type=float),
            'ndo_transient_attitude_boost_enabled': ParameterValue(ndo_transient_attitude_boost_enabled, value_type=bool),
            'ndo_transient_attitude_limit_deg': ParameterValue(ndo_transient_attitude_limit_deg, value_type=float),
            'ndo_transient_attitude_boost_duration': ParameterValue(ndo_transient_attitude_boost_duration, value_type=float),
            'ndo_transient_attitude_boost_fade': ParameterValue(ndo_transient_attitude_boost_fade, value_type=float),
            'ndo_base_gain_force': ParameterValue(ndo_base_gain_force, value_type=float),
            'ndo_base_gain_torque': ParameterValue(ndo_base_gain_torque, value_type=float),
            'ndo_lpf_alpha': ParameterValue(ndo_lpf_alpha, value_type=float),
            'ndo_attitude_gain': ParameterValue(ndo_attitude_gain, value_type=float),
            'ndo_adaptive_gain_enabled': ParameterValue(ndo_adaptive_gain_enabled, value_type=bool),
            'ndo_coupling_compensation_enabled': ParameterValue(ndo_coupling_compensation_enabled, value_type=bool),
            'inner_thrust_retarget_enabled': ParameterValue(inner_thrust_retarget_enabled, value_type=bool),
            'slider_command_shaping_enabled': ParameterValue(slider_command_shaping_enabled, value_type=bool),
            'slider_command_shape_tau': ParameterValue(slider_command_shape_tau, value_type=float),
            'slider_command_rate_limit': ParameterValue(slider_command_rate_limit, value_type=float),
            'outer_xy_kp': ParameterValue(outer_xy_kp, value_type=float),
            'outer_xy_ki': ParameterValue(outer_xy_ki, value_type=float),
            'outer_xy_kd': ParameterValue(outer_xy_kd, value_type=float),
            'outer_terminal_hover_velocity_damping': ParameterValue(outer_terminal_hover_velocity_damping, value_type=float),
            'body_mass_kg': ParameterValue(body_mass_kg, value_type=float),
            'moving_mass_kg': ParameterValue(moving_mass_kg, value_type=float),
            'slider_wn_mass': ParameterValue(slider_wn_mass, value_type=float),
            'slider_zeta_mass': ParameterValue(slider_zeta_mass, value_type=float),
            'actuation_backend': 'rotor_physics',
            'yaw_control_mode': 'rotor_only',
            'rotor_max_speed_rad_s': ParameterValue(rotor_max_speed_rad_s, value_type=float),
            'rotor_motor_time_constant_up': ParameterValue(rotor_motor_time_constant_up, value_type=float),
            'rotor_motor_time_constant_down': ParameterValue(rotor_motor_time_constant_down, value_type=float),
            'rotor_motor_rate_limit_rad_s2': ParameterValue(rotor_motor_rate_limit_rad_s2, value_type=float),
            'rotor_min_speed_ratio': 0.20,
            'rotor_tau_z_filter_time_constant': 0.25,
            'rotor_lower_command_scale': 1.0,
            'rotor_lower_command_scale_after_hover': 1.0,
            'rotor_lower_command_scale_after_hover_hold_time': 0.0,
            'outer_z_thrust_scale': 0.977,
            'attitude_step_test_enabled': False,
            'auto_scene_mode': ParameterValue(auto_scene_mode, value_type=str),
            'auto_scene_takeoff_transition_time': 5.0,
            'auto_scene_hover_hold_time': ParameterValue(auto_scene_hover_hold_time, value_type=float),
            'auto_scene_move_duration': ParameterValue(auto_scene_move_duration, value_type=float),
            'auto_scene_horizontal_accel_limit': ParameterValue(auto_scene_horizontal_accel_limit, value_type=float),
            'auto_scene_target_x': ParameterValue(auto_scene_target_x, value_type=float),
            'auto_scene_target_y': ParameterValue(auto_scene_target_y, value_type=float),
            'auto_scene_target_z': ParameterValue(auto_scene_target_z, value_type=float),
            'auto_scene_figure_eight_x_amplitude': 1.8,
            'auto_scene_figure_eight_y_amplitude': 1.8,
            'auto_scene_figure_eight_z_amplitude': 0.675,
            'auto_scene_figure_eight_forward_tilt_deg': 45.0,
            'auto_scene_figure_eight_period': 16.0,
            'auto_scene_figure_eight_ramp_duration': 5.0,
            'auto_scene_figure_eight_entry_phase_ratio': 1.0,
            'auto_scene_adaptive_phase_enabled': True,
            'auto_scene_adaptive_phase_min_rate': 0.45,
            'auto_scene_adaptive_phase_filter_time_constant': 0.15,
            'auto_scene_adaptive_phase_along_track_window': 1.00,
            'auto_scene_adaptive_phase_cross_track_window': 0.70,
            'auto_scene_adaptive_phase_speed_floor': 0.25,
            'auto_scene_adaptive_phase_position_floor': 0.25,
            'auto_scene_adaptive_phase_velocity_floor': 0.25,
            'auto_scene_adaptive_phase_lag_weight': 0.95,
            'auto_scene_adaptive_phase_cross_track_weight': 1.15,
            'auto_scene_adaptive_phase_velocity_weight': 0.30,
            'auto_scene_adaptive_phase_projection_align_time_constant': 0.30,
            'auto_scene_adaptive_phase_projection_deadband': 0.02,
            'auto_scene_adaptive_phase_projection_max_correction': 0.10,
            'auto_scene_yaw_ref_mode': ParameterValue(auto_scene_yaw_ref_mode, value_type=str),
            'auto_scene_yaw_ref_speed_floor': 0.05,
            'auto_scene_yaw_ref_rate_limit_deg_s': 25.0,
            'rviz_trajectory_enabled': True,
            'rviz_trajectory_frame_id': 'mmc_world',
            'rviz_actual_path_topic': '/mmc/trajectory/actual',
            'rviz_reference_path_topic': '/mmc/trajectory/reference',
            'rviz_actual_path_max_points': 5000,
            'rviz_reference_path_dt': 0.05,
            'rviz_reference_path_cycles': 1.0,
            'rviz_vehicle_marker_enabled': True,
            'rviz_vehicle_marker_topic': '/mmc/vehicle_marker',
            'rviz_vehicle_sphere_diameter': 0.09,
            'rviz_vehicle_arrow_length': 0.16,
            'rviz_vehicle_arrow_shaft_diameter': 0.0175,
            'rviz_vehicle_arrow_head_diameter': 0.04,
            'rviz_vehicle_arrow_z_offset': 0.015,
            'auto_scene_yaw_step_deg': ParameterValue(auto_scene_yaw_step_deg, value_type=float),
            'auto_scene_yaw_ramp_duration': ParameterValue(auto_scene_yaw_ramp_duration, value_type=float),
        },],
        output='screen'
    )

    # 8. 启动键盘平移控制节点（弹出独立键盘窗口）
    start_keyboard_teleop = Node(
        package='mmc_control',
        executable='keyboard_teleop_node',
        parameters=[{'input_backend': 'tk'}],
        output='screen'
    )

    # 按顺序和延时组装启动列表
    actions = [
        DeclareLaunchArgument('uav_model_path', default_value=urdf_path),
        DeclareLaunchArgument('world_sdf_path', default_value=default_world_path),
        DeclareLaunchArgument('enable_gui', default_value='true'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        DeclareLaunchArgument('enable_plotjuggler', default_value='false'),
        DeclareLaunchArgument('enable_wind_bridge', default_value='false'),
        DeclareLaunchArgument('ndo_enabled', default_value='true'),
        DeclareLaunchArgument('ndo_compensation_limit_deg', default_value='20.0'),
        DeclareLaunchArgument('ndo_compensation_limit_schedule_enabled', default_value='false'),
        DeclareLaunchArgument('ndo_compensation_limit_low_speed', default_value='3.0'),
        DeclareLaunchArgument('ndo_compensation_limit_high_speed', default_value='5.0'),
        DeclareLaunchArgument('ndo_compensation_limit_low_deg', default_value='12.0'),
        DeclareLaunchArgument('ndo_compensation_limit_high_deg', default_value='18.0'),
        DeclareLaunchArgument('ndo_compensated_attitude_limit_deg', default_value='25.0'),
        DeclareLaunchArgument('ndo_feedback_relief_enabled', default_value='false'),
        DeclareLaunchArgument('ndo_feedback_relief_gain', default_value='1.0'),
        DeclareLaunchArgument('ndo_feedback_relief_deadband_deg', default_value='1.5'),
        DeclareLaunchArgument('ndo_feedback_relief_max_fraction', default_value='0.65'),
        DeclareLaunchArgument('ndo_transient_attitude_boost_enabled', default_value='false'),
        DeclareLaunchArgument('ndo_transient_attitude_limit_deg', default_value='28.0'),
        DeclareLaunchArgument('ndo_transient_attitude_boost_duration', default_value='4.5'),
        DeclareLaunchArgument('ndo_transient_attitude_boost_fade', default_value='1.0'),
        DeclareLaunchArgument('ndo_base_gain_force', default_value='15.0'),
        DeclareLaunchArgument('ndo_base_gain_torque', default_value='15.0'),
        DeclareLaunchArgument('ndo_lpf_alpha', default_value='0.4'),
        DeclareLaunchArgument('ndo_attitude_gain', default_value='10.0'),
        DeclareLaunchArgument('ndo_adaptive_gain_enabled', default_value='true'),
        DeclareLaunchArgument('ndo_coupling_compensation_enabled', default_value='true'),
        DeclareLaunchArgument('inner_thrust_retarget_enabled', default_value='true'),
        DeclareLaunchArgument('slider_command_shaping_enabled', default_value='false'),
        DeclareLaunchArgument('slider_command_shape_tau', default_value='0.06'),
        DeclareLaunchArgument('slider_command_rate_limit', default_value='0.45'),
        DeclareLaunchArgument('outer_xy_kp', default_value='2.20'),
        DeclareLaunchArgument('outer_xy_ki', default_value='0.18'),
        DeclareLaunchArgument('outer_xy_kd', default_value='0.85'),
        DeclareLaunchArgument('outer_terminal_hover_velocity_damping', default_value='1.6'),
        DeclareLaunchArgument('body_mass_kg', default_value='1.25'),
        DeclareLaunchArgument('moving_mass_kg', default_value='0.05'),
        DeclareLaunchArgument('slider_wn_mass', default_value='20.0'),
        DeclareLaunchArgument('slider_zeta_mass', default_value='1.20'),
        DeclareLaunchArgument('rotor_max_speed_rad_s', default_value='450.0'),
        DeclareLaunchArgument('rotor_motor_time_constant_up', default_value='0.10'),
        DeclareLaunchArgument('rotor_motor_time_constant_down', default_value='0.16'),
        DeclareLaunchArgument('rotor_motor_rate_limit_rad_s2', default_value='0.0'),
        DeclareLaunchArgument('auto_scene_mode', default_value='hover_to_point_hold'),
        DeclareLaunchArgument('auto_scene_hover_hold_time', default_value='3.0'),
        DeclareLaunchArgument('auto_scene_move_duration', default_value='5.0'),
        DeclareLaunchArgument('auto_scene_horizontal_accel_limit', default_value='0.9'),
        DeclareLaunchArgument('auto_scene_target_x', default_value='0.0'),
        DeclareLaunchArgument('auto_scene_target_y', default_value='3.0'),
        DeclareLaunchArgument('auto_scene_target_z', default_value='2.0'),
        DeclareLaunchArgument('auto_scene_yaw_ref_mode', default_value='fixed'),
        DeclareLaunchArgument('auto_scene_yaw_step_deg', default_value='90.0'),
        DeclareLaunchArgument('auto_scene_yaw_ramp_duration', default_value='5.0'),
        set_env,
        start_gazebo_server,
        TimerAction(period=1.0, actions=[start_gazebo_gui]),
        TimerAction(period=3.0, actions=[spawn_model]),
        TimerAction(period=4.0, actions=[start_bridge]),
        TimerAction(period=5.0, actions=[start_rviz]),
        TimerAction(period=5.0, actions=[start_wind_bridge]),
        TimerAction(period=5.0, actions=[start_plotjuggler]),
        TimerAction(period=6.0, actions=[start_controller]),
    ]

    if manual_xy_enabled:
        actions.append(TimerAction(period=6.5, actions=[start_keyboard_teleop]))

    return LaunchDescription(actions)
