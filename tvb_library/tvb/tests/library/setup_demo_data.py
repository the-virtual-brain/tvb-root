import tempfile
from tvb.setup_demo_data import setup_data

def test_setup_demo_data():
    with tempfile.TemporaryDirectory() as path:
        setup_data(path)
