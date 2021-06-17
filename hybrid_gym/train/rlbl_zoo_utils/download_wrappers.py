# for maintenance purposes
import os
import urllib.request


def download_wrappers():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_dir, 'wrappers.py'), 'xb') as f:
        http = urllib.request.urlopen(
            'https://github.com/araffin/rl-baselines-zoo/raw/master/utils/wrappers.py'
        )
        f.write(http.read())


if __name__ == '__main__':
    download_wrappers()
