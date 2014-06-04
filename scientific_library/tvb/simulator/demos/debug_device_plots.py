import matplotlib as mpl
mpl.use('Agg')
from pylab import *
import pylab as pl
from numpy import *

try:
    ys1
except:
    d = load('debug.npz')
    ys1 = d['ys1']
    ys2 = d['ys2']
    dys1 = d['dys1']
    dys2 = d['dys2']

_y1, _y2, _dy1, _dy2 = [_y[:, 0, 0, ::32].T 
                         for _y in (ys1, ys2, dys1, dys2)]

pl.figure(1)
pl.clf()
pl.subplot(321)
for y in _y1:
    pl.plot(y, 'k-')
pl.title("X(t)")
pl.ylabel("Python")
pl.grid(True)
pl.xticks(pl.xticks()[0], ())

pl.subplot(323)
for y in _y2:
    pl.plot(y, 'k-')
pl.ylabel("CUDA")
pl.grid(True)
pl.xticks(pl.xticks()[0], ())

pl.subplot(322)
for y in _dy1:
    pl.plot(y, 'k-')
pl.title("d/dt X(t)")
pl.grid(True)
pl.xticks(pl.xticks()[0], ())

pl.subplot(324)
for y in _dy2:
    pl.plot(y, 'k-')
pl.grid(True)
pl.xticks(pl.xticks()[0], ())

pl.subplot(326)
for y1, y2 in zip(_dy1, _dy2):
    pl.plot(100*(y1-y2)/y1.ptp(), 'k-')
pl.xlabel('Time (ms)')
pl.grid(True)

pl.subplot(325)
for y1, y2 in zip(_y1, _y2):
    pl.plot(100*(y1-y2)/y1.ptp(), 'k-')
pl.xlabel('Time (ms)')
pl.grid(True)
pl.ylabel('% Rel. Error')

pl.tight_layout()

X, Y = [], []
for i, coupling_a in enumerate(r_[:0.1:16j]):
    for j, model_a in enumerate(r_[-2.0:2.0:16j]):
        X.append(model_a)
        Y.append(coupling_a)
X = array(X).reshape((16, 16))
Y = array(Y).reshape((16, 16))

pl.figure(2)
levels=r_[1.0 : 2.0 : 6j]
pl.clf()
pl.contour(X, Y, ys1[..., 0, :].std(0).mean(0).reshape((16, 16)), 
           levels, linestyles='dashed')
pl.contour(X, Y, ys2[..., 0, :].std(0).mean(0).reshape((16, 16)), 
           levels)
pl.ylabel('k')
pl.xlabel('a')
pl.title('Mean std dev (k, a)')
pl.grid(True)
pl.colorbar()
pl.tight_layout()
xt = r_[-2.:2.:16j]
xts = ['%.2f'%x if i%5==0 else '' for i, x in enumerate(xts)]
pl.xticks(xt, xts)
yt = r_[:0.1:16j]
yts = ['%.2f'%x if i%5==0 else '' for i, x in enumerate(yt)]
pl.yticks(yt, yts)
pl.show()




