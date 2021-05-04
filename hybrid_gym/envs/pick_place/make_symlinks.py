import site
import os

def make_symlinks():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    for package_dir in site.getsitepackages():
        gym_assets_dir = os.path.join(package_dir, 'gym', 'envs', 'robotics', 'assets')
        if os.path.isdir(gym_assets_dir):
            for subdir in ['stls', 'textures']:
                os.symlink(os.path.join(gym_assets_dir, subdir),
                           os.path.join(current_dir, subdir))
            for xml in ['robot.xml', 'shared.xml']:
                os.symlink(os.path.join(gym_assets_dir, 'fetch', xml),
                           os.path.join(current_dir, 'mujoco_xml', xml))
            return
    raise FileNotFoundError('failed to locate installed gym package')

if __name__ == '__main__':
    make_symlinks()
