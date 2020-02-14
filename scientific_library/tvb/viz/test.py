import numpy as np
import vtkplotter as vp
from tvb.simulator.lab import surfaces
vp.settings.useFXAA=True
cortex = surfaces.CorticalSurface.from_file()
cortex.configure()
cortex.compute_geodesic_distance_matrix(3)
gdm = cortex.geodesic_distance_matrix
gdm.data[:] = 1.0 / (1 - np.exp(-0.1*gdm.data))

mesh = vp.Mesh([cortex.vertices, cortex.triangles])
pt = vp.Plotter(axes=1)
pt += mesh
def do_slider(w, e):
    mesh.alpha(w.GetRepresentation().GetValue())
def do_color_slider(w, e):
    mesh.color(w.GetRepresentation().GetValue())
pt.addSlider2D(do_color_slider, -9, 9, value=0, pos=4, title="color number")
pt.addSlider2D(do_slider, xmin=0.01, xmax=0.99, value=0.5, pos=14, c="blue", title="alpha value")
pt += vp.Text2D("Cortical Surface with alpha")
mesh.addScalarBar3D(c="k", title="colorbar").scale(100)
mesh.phong()
diffuse = [0.1]
specular = [0.0]
def do_diffuse(w, e):
    diffuse[0] = w.GetRepresentation().GetValue()
    mesh.lighting('default', 0.1, diffuse[0], specular[0], 20, 'white')
def do_specular(w, e):
    specular[0] = w.GetRepresentation().GetValue()
    mesh.lighting('default', 0.1, diffuse[0], specular[0], 20, 'white')
pt.addSlider2D(do_diffuse, xmin=0.01, xmax=0.99, value=0.5, pos=12, c="yellow", title="diffuse value")
pt.addSlider2D(do_specular, xmin=0.01, xmax=0.99, value=0.5, pos=13, c="green", title="specular value")
def do_kernel(w, e):
    l = w.GetRepresentation().GetValue()
    val = gdm.copy()
    val.data[:] = 1.0 / (1 - np.exp(-np.exp(l)*val.data))
    mesh.pointColors(np.array(val[0].todense())[0], cmap='bone')
pt.addSlider2D(do_kernel, xmin=-5.0, xmax=1.0, value=-1.0, pos=11, c="red", title="kernel value")

# add the lorenz lines
dt = 0.002
y = (25.0, -10.0, -7.0)  # Starting point (initial condition)
pts, cols = [], []

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

# pt += vp.Point(y, r=10, c="g") # end point
# pt += vp.Points(pts, r=3, c=cols)
pt += vp.Line(pts).off().addShadow(x=3) # only show shadow, not line
pt += vp.Line(pts).off().addShadow(z=-30)

pt.show()
# vp.screenshot()
