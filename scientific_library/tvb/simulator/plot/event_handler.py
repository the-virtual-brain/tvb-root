import matplotlib
matplotlib.use('nbagg')
import random, sys, time
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
sys.path.append('../lib')
from typecheck import typecheck
import topo

x = True
while x:
    time.sleep(1)
    if random.random() < 0.1:
        x =False


class EventLoop:
    def __init__(self):
        self.command = None
        self.status = None
        self.handlers = {'interrupt': self.handle_interrupt}
        self.resolution = 0.1

    def loop(self):
        self.command = 'loop'
        while self.command != 'stop'
            self.status = 'running'
            time.sleep(self.resolution)

    def start(self):
        self.command = 'run'
        try:
            self.loop()
        except KeyboardInterrupt:
            self.handle_event('interrupt')

    def stop(self):
        self.command = 'stop'

    @typecheck
    def add_handler(self, fn: callable, event: str):
        self.handlers[event]()

    def handle_interrupt(self):
        print('Stopping event loop ...')
        self.stop()

class Callbacks:
    def __init__(self):
        (figure, axes) = plt.subplots()
        axes.set_aspect(1)
        figure.canvas.mpl_connect('button_press', self.press)
        figure.canvas.mpl_connect('button_release', self.release)

    def start(self):
        plt.show()

    def press(self, event):
        self.start_time = time.time()

    def release(self,event):
        self.end_time = time.time()
        self.draw_click(event)

    def draw_click(self, event):
        size = 4 * (self.end_time - self.start_time) ** 2
        c1 = plt.Circle([event.xdata, event.ydata], 0.002)
        c2 = plt.Circle([event.xdata, event.ydata], 0.02 * size, alpha=0.2)
        event.canvas.figure.gca().add_artist(c1)
        event.canvas.figure.gca().add_artist(c2)
        event.canvas.figure.show()

cbs = Callbacks()
cbs.start()
