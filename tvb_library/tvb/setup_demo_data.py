import os
import urllib.request
import zipfile
import logging
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

def setup_data(path):

    try:
        import tvb_data
        logger.info('tvb_data already available, quitting early.')
        return
    except:
        pass

    if not os.path.exists("tvb_data.zip"):
        urllib.request.urlretrieve(
            "https://zenodo.org/record/4263723/files/tvb_data.zip?download=1",
            os.path.join(path, "tvb_data.zip"))

    data_dir = os.path.join(path, "tvb_data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    with zipfile.ZipFile("tvb_data.zip") as zf:
        zf.extractall(data_dir)

    subprocess.check_call(
        [sys.executable, 'setup.py', 'install'],
        cwd=data_dir
    )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        _, path = sys.argv
        setup_data(path)
    else:
        with tempfile.TemporaryDirectory() as path:
            setup_data(path)