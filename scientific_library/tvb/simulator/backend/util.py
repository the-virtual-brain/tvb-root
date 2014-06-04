import time

class timer():
    def __init__(self):
        self.elapsed = 0.0
    def __enter__(self, t=time.time):
        self.tic = t()
    def __exit__(self, y, v, b, t=time.time):
        self.elapsed += t() - self.tic

if __name__ == '__main__':
    t = timer()
    n = int(1e6)
    for i in xrange(n):
        with t:
            pass
    print n, 'calls', t.elapsed
