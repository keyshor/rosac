# for maintenance purposes
import site
import os
import shutil


def duplicate_assets():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    for package_dir in site.getsitepackages():
        gym_env_dir = os.path.join(
            package_dir, 'gym', 'envs', 'mujoco',
        )
        gym_assets_dir = os.path.join(
            package_dir, 'gym', 'envs', 'mujoco', 'assets',
        )
        try:
            shutil.copyfile(os.path.join(gym_env_dir, 'ant_v3.py'),
                            os.path.join(current_dir, 'gym_ant.py'))
        except FileNotFoundError:
            print('failed to locate environment file')
        try:
            shutil.copyfile(os.path.join(gym_assets_dir, 'ant.xml'),
                            os.path.join(current_dir, 'gym_ant.xml'))
        except FileNotFoundError:
            print('failed to locate xml file')


if __name__ == '__main__':
    duplicate_assets()
