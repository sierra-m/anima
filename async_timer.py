import time


class Timer:
    def __init__(self, timeout_sec):
        self.timeout = timeout_sec
        self.start = time.time()

    def done(self):
        if (time.time() - self.start) >= self.timeout:
            self.start = time.time()
            return True
        return False

    def __bool__(self):
        return self.done()
