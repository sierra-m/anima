
class RGBColor:
    def __init__(self, item):
        """Constructor

        Opted for an array-based storage so color values can be negative
        """
        if type(item) is str:
            self.vals = self.expand(int(item.replace('#', ''), 16))
        elif type(item) is list or type(item) is tuple:
            self.vals = item
        else:
            self.vals = self.expand(item)

    def get_red(self):
        return self.vals[0]

    def get_green(self):
        return self.vals[1]

    def get_blue(self):
        return self.vals[2]

    @staticmethod
    def expand(val):
        return [(val >> 16) & 0xff, (val >> 8) & 0xff, val & 0xff]

    def compress(self):
        return ((self.vals[0] & 0xff) << 16) | ((self.vals[1] & 0xff) << 8) | (self.vals[2] & 0xff)

    def __int__(self):
        return self.compress()

    def __str__(self):
        return f'#{int(self)}{self.vals}'

    def __iter__(self):
        return iter(self.vals)

    def scale(self, scalar):
        scaled = [x * scalar for x in self.vals]
        return type(self)(scaled)

    def div(self, divisor):
        quotient = [x / divisor for x in self.vals]
        return type(self)(quotient)

    def __add__(self, other):
        added = [x + y for x, y in zip(self.vals, other.vals)]
        return type(self)(added)

    def __sub__(self, other):
        subbed = [x - y for x, y in zip(self.vals, other.vals)]
        return type(self)(subbed)

    def __round__(self, n=None):
        rounded = [round(x, n) for x in self.vals]
        return type(self)(rounded)
