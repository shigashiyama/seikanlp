import time

class Timer(object):
    def __init__(self):
        self.reset()


    def start(self):
        self.started = time.time()


    def stop(self):
        self.stopped = time.time()
        elapsed = self.stopped - self.started
        self.elapsed += elapsed


    def reset(self):
        self.started = 0
        self.stopped = 0
        self.elapsed = 0


    def restart(self):
        self.reset()
        self.start()


if __name__ == '__main__':
    timer = Timer()

    print('start')
    timer.start()
    for i in range(10000):
        pass
    timer.stop()
    print('stop / start')
    timer.start()
    for i in range(10000):
        pass
    timer.stop()
    print('stop')
    print(timer.elapsed)


