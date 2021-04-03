import os
import numpy
from numpy.ctypeslib import ndpointer
import ctypes as ct
from dataclasses import dataclass
from enum import Enum

from .templates import MakoUtilMix


class mathlib(Enum):
    default = 'default'
    fast = 'fast'
    svml = 'svml'
    system = 'system'


class target(Enum):
    sse2_i32x4 = 'sse2-i32x4'
    sse2_i32x8 = 'sse2-i32x8'
    sse4_i8x16 = 'sse4-i8x16'
    sse4_i16x8 = 'sse4-i16x8'
    sse4_i32x4 = 'sse4-i32x4'
    sse4_i32x8 = 'sse4-i32x8'
    avx1_i32x4 = 'avx1-i32x4'
    avx1_i32x8 = 'avx1-i32x8'
    avx1_i32x16 = 'avx1-i32x16'
    avx1_i64x4 = 'avx1-i64x4'
    avx2_i8x32 = 'avx2-i8x32'
    avx2_i16x16 = 'avx2-i16x16'
    avx2_i32x4 = 'avx2-i32x4'
    avx2_i32x8 = 'avx2-i32x8'
    avx2_i32x16 = 'avx2-i32x16'
    avx2_i64x4 = 'avx2-i64x4'
    avx512knl_i32x16 = 'avx512knl-i32x16'
    avx512skx_i32x8 = 'avx512skx-i32x8'
    avx512skx_i32x16 = 'avx512skx-i32x16'
    avx512skx_i8x64 = 'avx512skx-i8x64'
    avx512skx_i16x32 = 'avx512skx-i16x32'
    neon_i8x16 = 'neon-i8x16'
    neon_i16x8 = 'neon-i16x8'
    neon_i32x4 = 'neon-i32x4'
    neon_i32x8 = 'neon-i32x8'


@dataclass
class Options:
    "Options for the ISPC compiler."
    math_lib: mathlib = mathlib.default
    target: target = target.avx2_i32x8
    # TODO other options as required


class ISPCBackend(MakoUtilMix):

    @property
    def here(self) -> str:
        "Returns full path to current file."
        return os.path.dirname(os.path.abspath(__file__))

    _ispc = None
    @property
    def ispc(self) -> str:
        "Returns full to the ISPC compiler."
        if self._ispc is None:
            self._ispc = self.find_ispc()
        return self._ispc

    def find_ispc(self) -> str:
        "Returns the first ISPC compiler found on $PATH."
        for path in os.environ['PATH'].split(os.path.pathsep):
            for item in os.listdir(path):
                if item == 'ispc':
                    ispc = os.path.join(path, item)
                    logger.info('found ispc at %r', ispc)
                    return ispc

    def compile(self, fname: src, options: Options):
        "Compile a source file and return name of object file."

    

def _find_ispc():




def create_ispc_object(fname=None):
    cmd = f"{where_ispc()}
    subprocess.check_call(cmd)


def cmd(str):# {{{
    subprocess.check_call([_ for _ in str.split(' ') if _])# }}}

def make_kernel(): #{{{
    cxx = os.environ.get('CXX', 'g++')
    cxxflags = os.environ.get('CXXFLAGS', '')
    # cmd(cxx + ' -fPIC ' + cxxflags + ' -O3 -c network.ispc.cpp -o network.ispc.o')
    cmd('ispc --pic network.ispc -o network.ispc.o')
    cmd(cxx + ' -fPIC ' + cxxflags + ' -O3 -c tasksys.cpp -o tasksys.cpp.o')
    cmd(cxx + ' -shared tasksys.cpp.o network.ispc.o -o network.so')

    # need to break out the CMake build examples to figure out the right link
    # flags. Even without multicore, it looks a lot slower (8 it/s for 8 cores?)
    # cl tasksys.cpp??
    # cl /EHsc /c tasksys.cpp
    # cl /LD tasksys.obj
    #cmd('ispc --dllexport --math-lib=fast network.ispc -o network.obj')
    #cmd('lib /OUT:networkall.obj tasksys.obj network.obj')
    #cmd('link /DLL /NOENTRY /DEFAULTLIB:MSVCRT /export:integrate /OUT:networkall.dll networkall.obj')
    dll = ctypes.CDLL('./network.so')
    fn = getattr(dll, 'integrate')
    fn.restype = ctypes.c_void_p
    uint = ctypes.c_uint
    f32 = ctypes.c_float
    f32a = ctypes.POINTER(f32)
    fn.argtypes = [uint, uint, uint, uint, uint, f32, f32a, f32a, f32a, f32a, f32a, f32a]
    def _(*args):
        args_ = []
        for ct, arg_ in zip(fn.argtypes, args):
            if hasattr(arg_, 'shape'):
                args_.append(arg_.ctypes.data_as(ct))
            else:
                args_.append(ct(arg_))
        return fn(*args_)
    return _#}}}

def cf(array):#{{{
    # coerce possibly mixed-stride, double precision array to C-order single precision
    return array.astype(dtype='f', order='C', copy=True)#}}}

def nbytes(data):#{{{
    # count total bytes used in all data arrays
    nbytes = 0
    for name, array in data.items():
        nbytes += array.nbytes
    return nbytes#}}}

def make_gpu_data(data):#{{{
    # put data onto gpu
    gpu_data = {}
    for name, array in data.items():
        gpu_data[name] = gpuarray.to_gpu(cf(array))
    return gpu_data#}}}

# python parsweep.py -n 40 -b 32 -g 2
# 1.425 M step/s | 18.772 / 0.361 GB/s R/W | 23 Gops/s
def parse_args():#{{{
    parser = argparse.ArgumentParser(description='Run parameter sweep.')
    parser.add_argument('-b', '--block-dim', help='block dimension (default 32)', default=32, type=int)
    parser.add_argument('-g', '--grid-dim', help='grid dimensions (default 32)', default=32, type=int)
    parser.add_argument('-t', '--test', help='check results', action='store_true')
    parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=400)
    parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true')
    parser.add_argument('-p', '--no_progress_bar', help='suppress progress bar', action='store_false')
    parser.add_argument('--caching',
            choices=['none', 'shared', 'shared_sync', 'shuffle'],
            help="caching strategy for j_node loop (default shuffle)",
            default='shuffle'
            )
    parser.add_argument('--lineinfo', default=False, action='store_true')
    return parser.parse_args()#}}}


if __name__ == '__main__':

    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger('[parsweep_cuda]')
    logger.info('caching strategy %r', args.caching)

    # setup data#{{{
    weights, lengths = load_connectome()
    nc = args.block_dim
    ns = args.grid_dim
    logger.info('single connectome, %d x %d parameter space', ns, nc)
    logger.info('%d total num threads', ns * nc)
    couplings, speeds = setup_params(nc=nc, ns=ns)
    params_matrix = expand_params(couplings, speeds)#}}}

    # dimensions#{{{
    dt, tavg_period = 1.0, 1000.0
    # TODO buf_len per speed/block
    min_speed = speeds.min()
    n_work_items, n_params = params_matrix.shape
    n_nodes = weights.shape[0]
    buf_len_ = (lengths / min_speed / dt).astype('i').max() + 1
    buf_len = 2**np.argwhere(2**np.r_[:30] > buf_len_)[0][0]  # use next power of 2
    assert buf_len == 256  # FIXME
    logger.info('real buf_len %d, using power of 2 %d', buf_len_, buf_len)
    n_inner_steps = int(tavg_period / dt)#}}}

    # setup data#{{{
    data = { 'weights': weights, 'lengths': lengths,
            'couplings': couplings, 'speeds': speeds }
    base_shape = n_work_items,
    for name, shape in dict(
            tavg=(n_nodes,),
            state=(buf_len, n_nodes),
            ).items():
        data[name] = np.zeros(shape + base_shape, 'f')
    # make sure all C order and float32
    for key in list(data.keys()):
        data[key] = data[key].copy().astype(np.float32)
    logger.info('history shape %r', data['state'].shape)
    logger.info('on device mem: %.3f MiB' % (nbytes(data) / 1024 / 1024, ))#}}}

    #  compile kernels#{{{
    step_fn = make_kernel()#}}}

    # run simulation#{{{
    tic = time.time()
    nstep = args.n_time # 4s
    tavg = []
    for i in tqdm.tqdm(range(nstep)):
        step_fn(i*n_inner_steps, n_nodes, n_inner_steps, nc, ns, dt,
                data['speeds'], data['weights'], data['lengths'],
                data['couplings'], data['state'], data['tavg'])
        tavg.append(data['tavg'].copy())
    elapsed = time.time() - tic
    tavg = np.array(tavg)
    logger.info('elapsed time %0.3f', elapsed)
    step_total = nstep * n_inner_steps * n_work_items
    nn = 34
    step_read = 4*(1+n_inner_steps*(nn*(1 + nn*(1+1+1) + 1)))
    step_write = int(4*(nn/n_inner_steps+n_inner_steps*nn*2))
    step_ops = 2+2+nn*(2+1+nn*(2+1+1+1+1+5+4)+8+4)
    mbps_read = step_read / n_inner_steps * step_total / elapsed /(1024*1024*1024)
    mbps_write = step_write / n_inner_steps * step_total / elapsed / (1024*1024*1024)
    opss = step_ops * step_total / elapsed / (1024*1024*1024)
    logger.info('%0.3f M step/s | %0.3f / %0.3f GB/s R/W | %d Gops/s', 1e-6 * step_total / elapsed, mbps_read, mbps_write, opss)#}}}

    # check results (for smaller sizes)#{{{
    if args.test:
        r, c = np.triu_indices(n_nodes, 1)
        win_size = 200 # 2s
        win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
        err = np.zeros((len(win_tavg), n_work_items))
        # TODO do cov/corr in kernel
        for i, tavg_ in enumerate(win_tavg):
            for j in range(n_work_items):
                fc = np.corrcoef(tavg_[:, :, j].T)
                err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()
        # look at 2nd 2s window (converges quickly)
        err_ = err[1].reshape((speeds.size, couplings.size))
        # change on fc-sc metric wrt. speed & coupling strength
        derr_speed = np.diff(err_.mean(axis=1)).sum()
        derr_coupl = np.diff(err_.mean(axis=0)).sum()
        logger.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
        assert derr_speed > 500.0
        assert derr_coupl < -1500.0
        logger.info('result OK')

# vim: sw=4 sts=4 ts=4 et ai
