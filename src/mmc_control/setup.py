from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'mmc_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='WT2028',
    maintainer_email='15669367687@163.com',
    description='ROS 2 simulation controller for a moving-mass-controlled coaxial UAV.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller_node = mmc_control.mmc_uav:main',
            'keyboard_teleop_node = mmc_control.keyboard_teleop:main',
            'drone_visualizer = mmc_control.drone_visualizer:main',
            'wind_bridge_node = mmc_control.wind_bridge_node:main',
        ],
    },
)
