import numpy as np
from vtkplotter import *
import zipfile
settings.useFXAA = True

def surfzip2mesh(path):
    with zipfile.ZipFile(path) as zf:
        with zf.open("vertices.txt") as fd:
            vtx = np.loadtxt(fd)
        with zf.open("triangles.txt") as fd:
            tri = np.loadtxt(fd)
    return Mesh([vtx, tri])


subcort_path = '/home/duke/retro/1-Processed/id001_bt/tvb/surface_subcort.vep.zip'
cort_path = '/home/duke/retro/1-Processed/id001_bt/tvb/surface_cort.vep.zip'

# need a cortices + subcortical surfaces
subcort_mesh = surfzip2mesh(subcort_path).wireframe().decimate(0.1).alpha(0.1)
cort_mesh = surfzip2mesh(cort_path).wireframe().decimate(0.1).alpha(0.05).c("k")

# roi balls
path = '/home/duke/retro/1-Processed/id001_bt/tvb/connectivity.vep.zip'
with zipfile.ZipFile(path) as zf:
    with zf.open('centres.txt') as fd:
        roi_xyz = np.loadtxt(fd, usecols=(1,2,3))

rois = Spheres(roi_xyz, r=2, c='b', alpha=0.2)
#
# seeg balls
path = '/home/duke/retro/1-Processed/id001_bt/elec/seeg.xyz'
seeg_xyz = np.loadtxt(path, usecols=(1,2,3))
seeg = Spheres(seeg_xyz, r=2, c='y', alpha=0.2)

# use ezh to paint anatomy for comparison
path = '/home/duke/retro/1-Processed/id001_bt/tvb/ez_hypothesis.vep.txt'
ezh = np.loadtxt(path)

path = '/home/duke/retro/1-Processed/id001_bt/elec/elec-CT_in_T1.nii.gz'
ct = load(path, threshold=1000).alpha(0.05).c("r")

show(
    [subcort_mesh, cort_mesh, rois, seeg,
     # ct
     ],
    axes=5,
    shape='1/1',
    at=1,
    sharecam=False
)

# now time series colored by electrode
# with vertical bar
time_series = np.random.randn(10, 1000)
t = np.r_[:100.0:1j*time_series.shape[1]]
lines = [Line(t, y + i*3, c='k', alpha=0.2) for i, y in enumerate(time_series)]
show(lines, shape='1/1', at=0, sharecam=False, axes=0)

interactive()


