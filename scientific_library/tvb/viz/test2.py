import numpy as np
import nibabel
import vtkplotter
vtkplotter.settings.useFXAA= True

path = '/home/duke/src/tvb-root'
t1 = nibabel.load(f'{path}/T1.mgz')
nibabel.save(nibabel.Nifti1Image(t1.get_data(), t1.affine), f'{path}/t1.nii')
vol = vtkplotter.load(f'{path}/t1.nii')
vtkplotter.printHistogram(vol, logscale=True)
vol.crop(back=0.2, left=0.2, top=0.2)
lego = vol.legosurface(vmin=100, cmap='seismic')


vtkplotter.show(lego, shape='1/1', at=0, sharecam=False)

# add the lorenz lines
dt = 0.002
y = (25.0, -10.0, -7.0)  # Starting point (initial condition)
pts, cols = [], []
ts = []

for t in np.linspace(0, 20, int(20 / dt)):
    # Integrate a funny differential equation
    dydt = np.array(
        [-8 / 3.0 * y[0] + y[1] * y[2],
         -10.0 * (y[1] - y[2]),
         -y[1] * y[0] + 28.0 * y[1] - y[2]]
    )
    y = y + dydt * dt

    c = np.clip([np.linalg.norm(dydt) * 0.005], 0, 1)[0]  # color by speed
    cols.append([c, 0, 1-c])
    pts.append(y)
    ts.append(ts)


# pt += vp.Point(y, r=10, c="g") # end point
# pt += vp.Points(pts, r=3, c=cols)
# vp.show(vp.Line(pts).off().addShadow(x=3) # only show shadow, not line
vtkplotter.show(vtkplotter.Line(pts), shape='1/1', at=1, sharecam=False)#.off().addShadow(z=-30)
vtkplotter.vtkio.screenshot()
vtkplotter.interactive()