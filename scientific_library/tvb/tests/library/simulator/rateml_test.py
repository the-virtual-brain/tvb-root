"""
Test for RateML module

.. moduleauthor:: Aaron Perez Martin <a.perez.martin@fz-juelich.de>

"""

import pytest, os, glob, itertools, numpy as np, re, argparse, subprocess, pickle
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.rateML import XML2model
from tvb.rateML.XML2model import RateML
from tvb.rateML.run.model_driver import *
from pathlib import Path
import tvb.simulator.models

from lems.model.model import Model


xmlModelTesting = "kuramoto.xml"
framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = os.path.join(framework_path, "generatedModels")
cuda_ref_path = os.path.join(generatedModels_path, "cuda_refs")
run_path = os.path.join(framework_path, "run")
dic_regex_mincount = {r'^__global':1,
                      r'^__device':1,
                      r'^__device__ float wrap_it_':1,
                      r'state\(\(\(':1,
                      r'state\(\(t':2,
                      r'tavg\(':1,
                      r'= params\(\d\)\;$':2}
def compiler_opts():
    opts = ['--ptxas-options=-v', '-maxrregcount=32', '-lineinfo']
    opts.append('-lineinfo')
    opts.append('-DWARP_SIZE=%d' % (32,))
    opts.append('-DBLOCK_DIM_X=%d' % (32,))
    opts.append('-DNH=%s' % ('nh',))

def compile_cuda_model(location, model_name):
    source_file = os.path.join(location, model_name + ".c")
    compiled = False
    with open(source_file, 'r') as f:
        mod_content = f.read().replace('M_PI_F', '%ff' % (np.pi,))

        # Compile model
        mod = SourceModule(mod_content, options=compiler_opts(), include_dirs=[], no_extern_c=True, keep=False)
        if mod is not None:
            compiled = True
    return compiled

def check_input_params():
    if len(sys.argv)>=2:
        sys.argv = sys.argv[1:]

def find_attributes(obj, pattern):
    dict_found = {}
    count = 0
    for k,v in obj.__dict__.items():
        if k.startswith(pattern):
            dict_found[k] = v
            count +=1
    return dict_found, count

#This line avoid passing script name as a parameters
check_input_params()
class TestRateML():
    models=["epileptor", "kuramoto", "montbrio", "oscillator", "rwongwang"]
    python_mods = ["python"]*len(models)
    languages = ["python", "cuda"]

    # Cuda Section
    #----------------
    def test_make_cuda_setup(self):
        driver = Driver_Execute(Driver_Setup())

        bx, by = driver.args.blockszx, driver.args.blockszy
        nwi = driver.n_work_items
        rootnwi = int(np.ceil(np.sqrt(nwi)))
        gridx = int(np.ceil(rootnwi / bx))
        gridy = int(np.ceil(rootnwi / by))

        assert gridx * gridy * bx * by >= nwi

    def test_make_cuda_data(self):
        data = {}
        n_times = 20
        data["serie"] = np.zeros(2, 'f')


        driver = Driver_Execute(Driver_Setup())
        gpu_data = driver.make_gpu_data(data)

        assert gpu_data["serie"].size == data["serie"].size

    def test_make_cuda_kernel(self):
        driver = Driver_Execute(Driver_Setup())
        step_fn = driver.make_kernel(source_file=driver.args.filename, warp_size=32, args=driver.args,
                           lineinfo=driver.args.lineinfo, nh=driver.buf_len)
        assert step_fn is not None and len(str(step_fn))>0

    # Model Driver Section
    # --------------------
    def test_check_parameters(self):
        driver = Driver_Execute(Driver_Setup())
        _, count = find_attributes(driver.args, "n_sweep_")
        assert count == 2
        assert driver.exposures == 2 and driver.states == 2

        n_work_items, n_params = driver.params.shape
        assert n_work_items == 16 and n_params == 2

    # Model Section
    # ----------------
    @pytest.mark.slow
    @pytest.mark.parametrize('model_name', models)
    def test_load_model(self, model_name):
        model, _, _, _, _ = RateML(model_name).load_model()
        assert model is not None

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name, language', itertools.product(["kuramoto"], ["python"]))
    def test_prep_model_bound(self, model_name, language):
        # python only
        _, svboundaries, _, _, _ = RateML(model_name, language=language).load_model()
        assert svboundaries

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name, language', itertools.product(["kuramoto"], ["cuda"]))
    def test_prep_model_coupling(self, model_name, language):
        # cuda only
        _, _, couplinglist, _, _ = RateML(model_name, language=language).load_model()
        assert len(couplinglist) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name, language', itertools.product(["kuramoto"], ["cuda"]))
    def test_prep_model_noise(self, model_name, language):
        # cuda only
        _, _, _, noise, nsig = RateML(model_name, language=language).load_model()
        assert noise and nsig

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name, language', itertools.product(["kuramoto"], ["cuda"]))
    def test_time_serie(self, model_name, language):
        RateML(model_filename=model_name, language=language, XMLfolder=XMLModel_path,
               GENfolder=generatedModels_path)  # .render()
        driver = Driver_Execute(Driver_Setup())
        driver.args.n_time = 100
        driver.args.verbose = True
        tavg0 = driver.run_simulation()
        assert pytest.approx(driver.compare_with_ref(tavg0), 0.00001) == 1

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name, language', itertools.product(models, languages))
    def test_convert_model(self, model_name, language):
        model_str, driver_str = RateML(model_filename=model_name,language=language,XMLfolder=XMLModel_path,
                                       GENfolder=generatedModels_path).render()
        if language == "python":
            assert len(model_str) >0 and driver_str is None
        if language == "cuda":
            assert len(model_str) >0 and driver_str is not None and len(driver_str) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name', models)
    def test_compile_cuda_models(self, model_name):
        assert compile_cuda_model(location=generatedModels_path, model_name=model_name)

    @pytest.mark.slow
    @pytest.mark.parametrize('model_name', models)
    def test_contentcheck_cuda_models(self, model_name):

        source_file = os.path.join(generatedModels_path, model_name + ".c")
        with open(source_file, "r") as f:
            lines = f.read()

            # trucate the file to avoid processing bold_update
            lines = lines.split("__global__ void bold_update")[0]

            pattern_model = r'^__global__ void ' + model_name + '\('
            assert len(re.findall(pattern=pattern_model, string=lines, flags=re.IGNORECASE + re.MULTILINE + re.DOTALL)) >0

            for regex, mincount in dic_regex_mincount.items():
                matches = re.findall(pattern=regex, string=lines, flags=re.IGNORECASE + re.MULTILINE + re.DOTALL)
                assert len(matches) >= dic_regex_mincount[regex]

    # Simulation Section
    # ----------------
    @pytest.mark.slow
    @pytest.mark.parametrize('model_name', ["kuramoto"])
    def test_simulation_cuda_models(self, model_name):

        ##Move model to the script location
        cmd = "cp " + os.path.join(generatedModels_path, model_name + ".c") + " " + run_path
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        out, err = process.communicate()
        assert len(out) == 0 and len(err) == 0

        n_steps = 4
        path = os.path.join(run_path, "model_driver.py")
        cmd = "python " + path + " --model " + model_name + " -n " + str(n_steps) + " -w"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        out, err = process.communicate()

        # Reading the simulation data
        tavg_file = open('tavg_data', 'rb')
        tavg_data = pickle.load(tavg_file)
        tavg_file.close()
        a, b, c, d = tavg_data.shape
        assert (a, b, c, d) == (4, 2, 68, 16)
