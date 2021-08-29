'''
Implements a class to time functions as a chronometer
'''
import time


class Chronometer():
    def __init__(self):
        self._reference_time = None
        self._total_time = 0
    
    def start(self):
        self._reference_time = time.time()
        self._total_time = 0
    
    def pause(self):
        self._total_time += (time.time() - self._reference_time)
        self._reference_time = None
    
    def resume(self):
        self._reference_time = time.time()
    
    def elapsed(self):
        if self._reference_time is None:
            return self._total_time
        else:
            extra_time = (time.time() - self._reference_time)
            return self._total_time + extra_time


if __name__ == '__main__':
    chrono = Chronometer()
    chrono.start()
    time.sleep(3)
    print(chrono.elapsed())
    chrono.pause()
    time.sleep(2)
    print(chrono.elapsed())
    chrono.resume()
    time.sleep(2)
    print(chrono.elapsed())
