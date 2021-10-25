"""
Test for RateML module

.. moduleauthor:: Aaron Perez Martin <a.perez.martin@fz-juelich.de>
.. moduleauthor:: Michiel van der Vlag <m.van.der.vlag@fz-juelich.de>

"""
import importlib

import pytest, os, itertools, numpy as np, re, sys
from tvb.rateML import XML2model
from tvb.rateML.XML2model import RateML
from tvb.simulator.models.base import Model
pycuda = pytest.importorskip("pycuda")
try:
	import pycuda.autoinit
	PYCUDA_OK = True
except ImportError:
	PYCUDA_OK = False
skip_cuda_if_not_avail = pytest.mark.skipif(
	not PYCUDA_OK, reason='import pycuda.autoinit failed, CUDA not available')

xmlModelTesting = "kuramoto.xml"
framework_path, _ = os.path.split(XML2model.__file__)
XMLModel_path = os.path.join(framework_path, "XMLmodels")
generatedModels_path = os.path.join(framework_path, "generatedModels")
cuda_ref_path = os.path.join(generatedModels_path, "cuda_refs")
run_path = os.path.join(framework_path, "run")
dic_regex_mincount = {r'^__global':1,
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
	import pycuda.autoinit
	from pycuda.compiler import SourceModule

	source_file = os.path.join(location, model_name + ".c")
	compiled = False
	with open(source_file, 'r') as f:
		mod_content = f.read().replace('M_PI_F', '%ff' % (np.pi,))
		# TODO replace lower pi and inf need to be fixed to whole words only
		# mod_content = mod_content.replace('pi', '%ff' % (np.pi,))
		# mod_content = mod_content.replace('inf', 'INFINITY')

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

def remove_model_driver(model):

	rm_model = generatedModels_path + '/' + model + '.c'
	rm_driver = run_path + '/model_driver.py'
	if os.path.exists(rm_model):
		os.remove(rm_model)
	if os.path.exists(rm_driver):
		os.remove(rm_driver)

def setup_namespace(model='kuramoto'):

	# remove files to be sure there is always a new version
	remove_model_driver(model)

	# gemerate model and setup namespace for every test
	RateML(model, 'cuda')
	driver = importlib.import_module('.model_driver_' + model, 'tvb.rateML.run')
	driver = driver.Driver_Execute(driver.Driver_Setup())

	return driver

#This line avoid passing script name as a parameters
check_input_params()
class TestRateML():
	models=["epileptor", "kuramoto", "montbrio", "oscillator", "rwongwang"]
	python_mods = ["python"]*len(models)
	languages = ["python", "cuda"]

	# Cuda Section
	#----------------
	@skip_cuda_if_not_avail
	def test_make_cuda_setup(self):

		driver = setup_namespace('oscillator')
		bx, by = driver.args.blockszx, driver.args.blockszy
		nwi = driver.n_work_items
		rootnwi = int(np.ceil(np.sqrt(nwi)))
		gridx = int(np.ceil(rootnwi / bx))
		gridy = int(np.ceil(rootnwi / by))

		assert gridx * gridy * bx * by >= nwi

	@skip_cuda_if_not_avail
	def test_make_cuda_data(self):
		data = {}
		data["serie"] = np.zeros(2, 'f')

		driver = setup_namespace('kuramoto')
		gpu_data = driver.make_gpu_data(data)

		assert gpu_data["serie"].size == data["serie"].size

	@skip_cuda_if_not_avail
	def test_make_cuda_kernel(self):
		driver = setup_namespace()
		step_fn = driver.make_kernel(source_file=driver.args.filename, warp_size=32, args=driver.args,
						   lineinfo=driver.args.lineinfo, nh=driver.buf_len)
		assert step_fn is not None and len(str(step_fn))>0

	# Model Driver Section
	# --------------------
	def test_check_parameters(self):

		# works for default workitems settings
		driver = setup_namespace('oscillator')
		_, count = find_attributes(driver.args, "n_sweep_")
		assert count == 2
		assert driver.exposures == 2
		assert driver.states == 2

		# 16 workitems is based on driver default
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
	def test_prep_model_bound(self):
		# python only
		_, svboundaries, _, _, _ = RateML("kuramoto", language="cuda").load_model()
		assert svboundaries

	@pytest.mark.slow
	def test_prep_model_noise(self):
		# works for xml models which have noise enabled and nsig defined.
		# cuda only
		_, _, _, noise, nsig = RateML('kuramoto', 'cuda').load_model()
		assert noise
		assert nsig

	@pytest.mark.slow
	def test_prep_model_coupling(self):
		# cuda only
		_, _, couplinglist, _, _ = RateML("oscillator", "cuda").load_model()
		assert len(couplinglist) > 0

	@pytest.mark.slow
	def test_time_serie(self):
		driver = setup_namespace('oscillator')
		driver.args.n_time = 150
		tavg0 = driver.run_simulation()
		assert np.allclose(driver.compare_with_ref(tavg0), 1, 1e-6, 1e-6)

	@pytest.mark.slow
	@pytest.mark.parametrize('model_name, language', itertools.product(models, languages))
	def test_convert_model(self, model_name, language):
		if language == 'cuda' and not PYCUDA_OK:
			return
		# reload(Driver_Execute)
		model_str, driver_str = RateML(model_filename=model_name,language=language,XMLfolder=XMLModel_path,
									   GENfolder=generatedModels_path).render()
		if language == "python":
			assert len(model_str) >0 and driver_str is None
		if language == "cuda":
			assert len(model_str) >0 and driver_str is not None and len(driver_str) > 0

	@skip_cuda_if_not_avail
	@pytest.mark.slow
	@pytest.mark.parametrize('model_name', models)
	def test_compile_cuda_models(self, model_name):
		assert compile_cuda_model(location=generatedModels_path, model_name=model_name)

	@skip_cuda_if_not_avail
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
	@skip_cuda_if_not_avail
	@pytest.mark.slow
	def test_simulation_cuda_model_osc(self):
		# simulate with different properties to check if output shape is related

		driver = setup_namespace('oscillator')
		driver.args.n_time = 4
		driver.exposures = 2
		driver.n_regions = 68
		driver.n_work_items = 8
		tavg_data = driver.run_simulation()

		# a = n_steps, b = n_expos, c = n_regions, d = n_workitems
		a, b, c, d = tavg_data.shape
		assert (a, b, c, d) == (4, 2, 68, 8)

	@skip_cuda_if_not_avail
	@pytest.mark.slow
	def test_simulation_cuda_model_kur(self):
		# simulate with different properties to check if output shape is related

		driver = setup_namespace('kuramoto')
		driver.args.n_time = 8
		driver.exposures = 1
		driver.args.n_regions = 76
		driver.n_work_items = 12
		tavg_data = driver.run_simulation()

		# a = n_steps, b = n_expos, c = n_regions, d = n_workitems
		a, b, c, d = tavg_data.shape
		assert (a, b, c, d) == (8, 1, 76, 12)

	@skip_cuda_if_not_avail
	@pytest.mark.slow
	def test_simulation_cuda_model_epi(self):
		# simulate with different properties to check if output shape is related

		driver = setup_namespace('epileptor')
		driver.args.n_time = 12
		driver.exposures = 6
		driver.args.n_regions = 96
		driver.n_work_items = 18
		tavg_data = driver.run_simulation()

		# a = n_steps, b = n_expos, c = n_regions, d = n_workitems
		a, b, c, d = tavg_data.shape
		assert (a, b, c, d) == (12, 6, 96, 18)

	def test_load_model(self):
		name = 'epileptor'
		model, _, _, _, _ = RateML(name).load_model()
		assert model is not None

	def test_render_model(self):
		name = 'epileptor'
		model_str, _ = RateML(name).render()
		assert '_numba_dfun_EpileptorT' in model_str

	# TODO fix
	# def test_eval_model_str(self):
	# 	filename = 'epileptor'
	# 	classname = 'EpileptorT'
	# 	module = {}
	# 	exec(RateML(filename).render()[0], module)
	# 	assert issubclass(module[classname], Model)
	# 	model = module[classname]()
	# 	assert isinstance(model, Model)

