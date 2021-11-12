import os

import pytest
from tvb.basic.config.settings import HPCSettings
from tvb.basic.profile import TvbProfile
from tvb.core.services.backend_clients.hpc_client import HPCClient
from tvb.core.services.backend_clients.hpc_pipeline_client import HPCPipelineClient
from tvb.tests.framework.core.factory import TestFactory


@pytest.mark.skip
def test_hpc_pipeline(operation_factory):
    test_user = TestFactory.create_user()
    test_project = TestFactory.create_project(test_user)

    op = operation_factory(test_user=test_user, test_project=test_project)

    # Add correct path to tvb_bin folder and a valid ebrains auth token
    os.environ[HPCClient.TVB_BIN_ENV_KEY] = "/tvb-root/tvb_bin"
    os.environ[HPCClient.CSCS_LOGIN_TOKEN_ENV_KEY] = ""
    TvbProfile.current.hpc.HPC_COMPUTE_SITE = 'DAINT-CSCS'

    hpc_pipeline_client = HPCPipelineClient()
    job = hpc_pipeline_client._launch_job_with_pyunicore(op)
    assert job is not None
