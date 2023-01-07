import board  # noqa
import neopixel  # noqa
from abc import ABC
from abc import abstractmethod
from perlin_noise import PerlinNoise
from async_timer import Timer
from color import RGBColor
from mqttsubscriber import MQTTSubscriber
from typing import List
from enum import Enum
import random


class RGBAColor(RGBColor):
    def __init__(self, val, alpha=1.0):
        if isinstance(val, str) and len(val) >= 8:
            val = val.replace('#', '')
            super().__init__(val[0:5])
            self.alpha = int(val[6:7], 16)
        else:
            super().__init__(val[0:5])
            self.alpha = alpha

    @classmethod
    def null(cls):
        return cls(0x000000, alpha=0.0)

    @classmethod
    def black(cls):
        return cls(0x000000)

    @classmethod
    def white(cls):
        return cls(0xffffff)

    def composite_over(self, other):
        alpha_out = self.alpha + other.alpha * (1 - self.alpha)

        color_a_partial = self.scale(self.alpha)
        color_b_partial = other.scale(other.alpha * (1 - self.alpha))

        color_out = (color_a_partial + color_b_partial).div(alpha_out)
        color_out.alpha = alpha_out

        return color_out

    def get_rgb_vals(self):
        return list(map(lambda x: int(x * 3), self.vals))


class Layer:
    def __init__(self, buffer: List[RGBAColor] = None, size: int = None):
        if buffer is None and size is None:
            raise ValueError('Must define either `buffer` or `size`')

        self.buffer = buffer if buffer is not None else [RGBAColor.null() for i in range(size)]

    def __iadd__(self, other):
        if type(other) is not RGBAColor:
            other = RGBAColor(other)
        self.buffer.append(other)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def __setitem__(self, key, value):
        if not isinstance(value, RGBAColor):
            value = RGBAColor(value)
        self.buffer[key] = value

    def __iter__(self):
        return iter(self.buffer)

    def rotate_up(self):
        self.buffer = self.buffer[-1:] + self.buffer[:-1]

    def rotate_down(self):
        self.buffer = self.buffer[1:] + self.buffer[:1]

    def shift_up(self, color: RGBAColor):
        if not color:
            color = RGBAColor.null()
        self.buffer = [color] + self.buffer[:-1]

    def shift_down(self, color: RGBAColor):
        if not color:
            color = RGBAColor.null()
        self.buffer = [color] + self.buffer[:-1]

    def composite_over(self, other):
        out = []  # type: List[RGBAColor]

        for i in range(len(self.buffer)):
            out.append(self.buffer[i].composite_over(other.buffer[i]))

        return type(self)(out)


class Drawing(ABC):
    def __init__(self, *args, **kwargs):
        self.layer = kwargs.get('layer')

        if not self.layer:
            raise ValueError('Drawing requires `layer` argument')

    @abstractmethod
    def draw(self):
        pass


class Paint(Drawing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_color = kwargs.get('from_color', RGBAColor.null())
        self.to_color = kwargs.get('to_color', RGBAColor.null())
        self.from_idx = kwargs.get('from_idx', 0)
        self.to_idx = kwargs.get('to_idx', 0)

        if self.to_idx < 0:
            self.to_idx = len(self.layer) + self.to_idx + 1

        # Reverse vals if negative direction
        if self.from_idx > self.to_idx:
            self.from_idx, self.to_idx = self.to_idx, self.from_idx
            self.from_color, self.to_color = self.to_color, self.from_color

        self.span = self.to_idx - self.from_idx

    def draw(self):
        color_diff = self.to_color - self.from_color
        color_step = color_diff.div(self.span)

        a_step = (self.to_color.alpha - self.from_color.alpha) / self.span

        print(f'color step: {color_step}')

        curr_color = self.from_color
        for i in range(self.from_idx, self.to_idx):
            self.layer[i] = round(curr_color)
            curr_color += color_step
            curr_color.alpha += a_step


class Stroke(Drawing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color = kwargs.get('color', RGBAColor.null())
        self.from_idx = kwargs.get('from_idx', 0)
        self.to_idx = kwargs.get('to_idx', 0)

        if self.to_idx < 0:
            self.to_idx = len(self.layer) + self.to_idx + 1

        # Reverse vals if negative direction
        if self.from_idx > self.to_idx:
            self.from_idx, self.to_idx = self.to_idx, self.from_idx

    def draw(self):
        for i in range(self.from_idx, self.to_idx):
            self.layer[i] = self.color


class Animation(ABC):
    def __init__(self, *args, **kwargs):
        self.layer = kwargs.get('layer')
        self.delay_ms = kwargs.get('delay_ms', 0)
        self.timer = Timer(self.delay_ms / 1000)

        if not self.layer:
            raise ValueError('Animation requires `layer` argument')

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def run_step(self):
        if self.timer:
            self.step()
            return True
        return False


class Rotate(Animation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotate_up = kwargs.get('rotate_up', True)

    def pre(self):
        pass

    def step(self):
        if self.rotate_up:
            self.layer.rotate_up()
        else:
            self.layer.rotate_down()


class TwinklePattern(Enum):
    PROP_UP = 0
    PROP_DOWN = 1
    RANDOM = 2


class Twinkle(Animation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern = kwargs.get('pattern', TwinklePattern.PROP_UP)
        self.velocity = kwargs.get('velocity', 0.1)

        layer_range = range(len(self.layer))
        offset = random.random()
        if self.pattern == TwinklePattern.PROP_DOWN:
            self.noise_seeds = [x + offset for x in layer_range]
        elif self.pattern == TwinklePattern.PROP_UP:
            self.noise_seeds = [x + offset for x in reversed(layer_range)]
        elif self.pattern == TwinklePattern.RANDOM:
            self.noise_seeds = [random.random() for x in layer_range]

        self.noise = PerlinNoise()
        self.pixel_refs = []

    @staticmethod
    def calc_noise_ratio(noise_val, max_val):
        return (noise_val + 0.5) * max_val

    def pre(self):
        self.pixel_refs = list(self.layer)

    def step(self):
        for i in range(len(self.layer)):
            noise_val = self.noise(self.noise_seeds[i])

            self.layer[i].alpha = noise_val

            self.noise_seeds[i] += self.velocity


class Viewport:
    def __init__(self, buffer_size, delay_ms: int = 100):
        self.view_from = 0
        self.view_to = buffer_size - 1
        self.size = buffer_size
        self.velocity = 0
        self.timer = Timer(delay_ms / 1000)

    def step(self):
        self.view_from = (self.view_from + self.velocity) % self.size
        self.view_to = (self.view_to + self.velocity) % self.size

    def set_direction(self, pan_up: bool):
        self.velocity = 1 if pan_up else -1

    def set_delay(self, delay_ms: int):
        self.timer.timeout = delay_ms / 1000

    def run_step(self):
        if self.timer:
            self.run_step()


class BracketException(SyntaxError):
    pass


class ArgumentException(BaseException):
    pass


class Anima:
    operation_sizes = {
        'STR': 3,
        'PNT': 4,
        'ROT': 2,
        'TWK': 3
    }

    twinkle_types = {
        'up': TwinklePattern.PROP_UP,
        'down': TwinklePattern.PROP_DOWN,
        'rand': TwinklePattern.RANDOM
    }

    def __init__(self, pin, size, brightness=0.7):
        self.pixels = neopixel.NeoPixel(pin, size, bpp=3, auto_write=False, brightness=brightness)

        self.size = size
        self.palette = []
        self.base_layer = Layer(size=size)  # Layer to composite upon
        self.layers = []
        self.animations = []

        self.viewport = Viewport(size)

    def execute(self, command: str):
        macro_tokens = self.tokenize(command)

        # Level 1 macros - palette, layer
        for token in macro_tokens:
            name, middle = self.split_command(token)
            if name == 'PAL':
                colors = self.tokenize(middle)
                self.palette = map(lambda x: RGBAColor(x), colors)
            elif name == 'LAY':
                new_layer = Layer(self.size)
                self.layers.append(new_layer)
                draw_cmds = self.tokenize(middle)

                for cmd in draw_cmds:
                    op_name, args_list = self.split_command(cmd)
                    args = self.tokenize(args_list)

                    if self.operation_sizes[op_name] != len(args):
                        raise ArgumentException(f'Incorrect argument length for {op_name}')

                    self.add_operation(new_layer, op_name, args)
            elif name == 'PAN':
                args = self.tokenize(middle)
                self.viewport.set_direction(pan_up=args[0].lower() == 'up')
                self.viewport.set_delay(delay_ms=int(args[1]))
                return
            else:
                raise ValueError('May not define draw/animation commands at top level')

    def add_operation(self, new_layer: Layer, op_name, args):
        operation = None
        if op_name == 'STR':
            operation = Stroke(layer=new_layer,
                               color=self.render_color(args[0]),
                               from_idx=int(args[1]),
                               to_idx=int(args[2]))
        elif op_name == 'PNT':
            operation = Paint(layer=new_layer,
                              from_color=self.render_color(args[0]),
                              to_color=self.render_color(args[1]),
                              from_idx=int(args[2]),
                              to_idx=int(args[3]))
        elif op_name == 'ROT':
            operation = Rotate(layer=new_layer,
                               rotate_up=1 if args[0].lower() == 'up' else 0,
                               delay_ms=int(args[1]))
        elif op_name == 'TWK':
            operation = Twinkle(layer=new_layer,
                                pattern=self.twinkle_types[args[0]] if args[
                                                                           0] in self.twinkle_types else TwinklePattern.PROP_UP,
                                velocity=float(args[1]),
                                delay_ms=int(args[2]))
        else:
            raise TypeError(f'No such operation {op_name}')

        if isinstance(operation, Drawing):
            operation.draw()
        else:
            operation.pre()
            self.animations.append(operation)

    def update(self):
        # iterate from base to top
        results = map(lambda anim: anim.run_step(), self.animations)

        # if any motion occurs, composite layers and assign
        if any(results):
            build_layer = self.base_layer

            for layer in self.layers:
                build_layer = layer.composite_over(build_layer)

            for i, color in enumerate(build_layer):
                self.pixels[i] = color.get_rgb_vals()

            self.pixels.show()

    def render_color(self, color: str):
        if len(color) >= 6:
            return RGBAColor(color)
        val = int(color)
        if val >= len(self.palette):
            raise ValueError(f'color #{color} out of palette bounds')
        return self.palette[val]

    @staticmethod
    def tokenize(command: str):
        paren_count = 0
        start = 0
        out = []
        for i, char in enumerate(command):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                out.append(command[start:i])
                start = i + 1
            if paren_count < 0:
                raise BracketException('Unbalanced brackets: too many closing')
        if paren_count > 0:
            raise BracketException('Unbalanced brackets: too many opening')

        out.append(command[start:])
        return [x for x in out if x]

    @staticmethod
    def split_command(command: str):
        name, sfx = command.split('(', maxsplit=1)
        middle = sfx.rsplit(')', maxsplit=1)[0]
        return name, middle
