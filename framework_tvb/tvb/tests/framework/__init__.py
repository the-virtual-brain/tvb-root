"""
This package contains tests for the modules located in TVB module.
"""

from tvb.basic.profile import TvbProfile
if TvbProfile.CURRENT_PROFILE_NAME not in [TvbProfile.TEST_SQLITE_PROFILE, TvbProfile.TEST_POSTGRES_PROFILE]:
    TvbProfile.set_profile(TvbProfile.TEST_SQLITE_PROFILE)

ADAPTERS = {"AdaptersTest": {'modules': ["tvb.tests.framework.adapters"], 'rawinput': True}}
DATATYPES_PATH = ["tvb.tests.framework.datatypes", "tvb.datatypes"]
REMOVERS_PATH = ["tvb.datatype_removers"]
PORTLETS_PATH = ["tvb.tests.framework.core.portlets"]