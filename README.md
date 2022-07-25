# Anima
Lightweight MicroPython animation library for NeoPixel strips

# Purpose
This library is designed to display complex animations on a 2D individually addressable LED strip from relatively simple string-based
instructions. Users can send instructions to the target board using MQTT, allowing any IoT device with access to control strip animations remotely.

The intended targets of this project are raspberry Pi and ESP boards, though it should work on any MicroPython instantiation.

This project is in development.

# Example

Below is a simple animation instruction:

```
"PAL(FF32AE,3338FF,000000),LAY(PNT(0,1,0,END),TWK(RAND,100)),LAY(STR(2,0,3),ROT(DOWN,10))"
```
- Set palette to:
  - #0 hot pink (FF32AE)
  - #1 cool blue (3338FF)
  - #2 black (000000)
- Create a layer:
  - Paint hot pink to cool blue across strip length
  - Twinkle with random pattern at 100ms/step
- Create a layer above:
  - Stroke black from 0 to 3
  - Rotate down at 10ms/step

# Instructions

Anima makes use of an optional palette, a layer system, drawing commands and animation commands. Animations are built in a layer-by-layer basis, 
with all drawing and animating commands taking place within a layer. Drawing commands occur first, in the order defined, and then animations follow acting upon 
the existing layer.
