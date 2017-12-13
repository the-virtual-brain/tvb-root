# content of conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--profile", action="store", default="TEST_POSTGRES_PROFILE",
        help="my option: TEST_POSTGRES_PROFILE or TEST_SQLITE_PROFILE")

@pytest.fixture
def profile(request):
    return request.config.getoption("--profile")