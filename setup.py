from setuptools import setup
from glob import glob
import os

package_name = 'tb3_pick_place'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.world')),
        (os.path.join('share', package_name, 'urdf'),
         ['urdf/turtlebot3_manipulation_grasp.urdf.xacro']),
        (os.path.join('share', package_name, 'yolomodel'),
         glob('yolomodel/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='midhun',
    maintainer_email='midhun@todo.todo',
    description='TB3 + OpenMANIPULATOR-X autonomous YOLO pick-and-place',
    license='Apache-2.0',
    tests_require=['pytest'],
    scripts=[
        'scripts/autonomous_pick_place.py',
        'capture_dock_data.py',
    ],
    entry_points={
        'console_scripts': [
            'pick_place_demo = tb3_pick_place.pick_place_demo:main',
            'yolo_pick_place = tb3_pick_place.main:main',
        ],
    },
)
