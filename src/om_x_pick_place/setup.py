import os
from glob import glob
from setuptools import setup

package_name = 'om_x_pick_place'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.com',
    description='OpenMANIPULATOR-X pick-and-place in Gazebo',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'pick_and_place = scripts.pick_and_place:main',
            'pick_place_direct = scripts.pick_place_direct:main',
        ],
    },
)
