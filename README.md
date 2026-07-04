# Acoustic Tracking and Binary Detection

Real-time acoustic **Direction-of-Arrival (DOA)** tracking and **tone detection** on a
motorized pan–tilt platform. A four-microphone array estimates the bearing of the
dominant sound source, and an STM32 microcontroller steers the platform to face it.
In parallel, the system performs **binary (present / absent) detection** of narrowband
tones.

> **Project 25-1-1-3234** — Iby and Aladar Fleischman Faculty of Engineering, Tel Aviv University.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Serial protocol](#serial-protocol)
- [Configuration](#configuration)
- [Results](#results)
- [Further work](#further-work)
- [Authors](#authors)
- [References](#references)

---

## Overview

The system is split into two cooperating parts that run on different hardware and
communicate over a serial link:

- A **perception tier**, written in Python on a host computer, which captures audio
  from the microphone array, detects the dominant tone, estimates its direction, and
  streams the bearing to the microcontroller.
- A **control tier**, written in C on an STM32 microcontroller, which receives the
  bearing and drives a pan–tilt rig so that the array points at the source.

Because the microphone array is mounted on the pan axis, turning the platform changes
what the array hears: the system is a closed loop, and the controller drives the
measured bearing toward zero (null-seeking).

The DOA algorithm can also be run entirely offline in a **simulation mode**, where the
microphone signals for a known source direction are synthesised mathematically, so the
estimator can be validated against an exact ground truth.

## Key features

- **GCC-PHAT** time-delay estimation across microphone pairs, robust to reverberation.
- **Direction solve** combining the pairwise delays into an azimuth/elevation bearing,
  followed by **exponential-moving-average (EMA)** smoothing.
- **Binary tone detection** using a locally adaptive (CFAR-style) spectral threshold and
  a **hysteresis counter** for temporal persistence — confirming a tone only once it is
  sustained, and rejecting transient noise.
- **Velocity-mode PID** motor control with anti-windup, derivative limiting, slew-rate
  limiting, and shortest-path angle wrapping.
- **Race-free serial reception** on the microcontroller via a double-buffered
  (ping-pong) interrupt service routine, with a heartbeat timeout that parks the
  platform if the link is lost.
- **Built-in simulation harness** with circle / arc / spiral test trajectories and
  automatic error statistics (RMSE, MAE, max error).

## How it works

```
 Microphone array ──USB──▶  Host PC  (Python, perception tier)
                            ├─ tone detection  (CFAR + hysteresis) ─▶ f*
                            ├─ band-pass around f*
                            ├─ GCC-PHAT time-delay per mic pair
                            ├─ direction solve  (azimuth, elevation)
                            └─ EMA smoothing
                                     │
                                     │  UART, 115200 baud   "P:<phi>,T:<theta>\n"
                                     ▼
                            STM32  (C, control tier)
                            ├─ UART RX  (double-buffered ISR)
                            ├─ velocity-mode PID  (pan / tilt)
                            └─ L6474 stepper drivers (SPI)
                                     │
                                     ▼
                            Pan / tilt steppers + rig
                                     │
              rig carries the array  │  (closed loop)
                     ◀───────────────┘
```

## Hardware

| Component | Description |
|---|---|
| Microphone array | ReSpeaker USB Mic Array — four MEMS microphones in a circular layout |
| Host computer | Any PC with a USB port and a serial connection to the microcontroller |
| Microcontroller | STM32 Nucleo development board *(confirm exact part number)* |
| Motor driver | X-NUCLEO-IHM01A1 stepper-driver expansion (L6474), one driver per axis |
| Actuators | Two stepper motors (pan and tilt) on a pan–tilt rig carrying the array |

> Pan and tilt use different gear ratios (≈ 3345 microsteps/rev on pan, 16576 on tilt),
> which the firmware accounts for in its step-to-angle conversion.

## Repository structure

> Adjust to match your actual repository; the Python module names below are the ones
> used in the source.

```
.
├── perception/                 # Python perception tier
│   ├── main.py                 # entry point (live or simulation)
│   ├── micArray.py             # acquisition, simulation, visualisation, statistics
│   ├── signal_processing.py    # GCC-PHAT, direction solve, smoothing
│   ├── constants.py            # array geometry, sample rate, thresholds, tuning
│   ├── robotics.py             # serial interface to the microcontroller
│   └── gui/                    # supervising GUI  (fill in)
├── firmware/                   # STM32 control tier (C)
│   ├── Core/                   # main.c (UART ISR + velocity-mode PID), HAL config
│   ├── Drivers/                # STM32 HAL + L6474 / motor-control BSP
│   └── *.ioc                   # STM32CubeIDE project
├── docs/                       # project report and figures
└── README.md
```

## Requirements

**Perception tier (Python 3):**

- `numpy`
- `scipy`
- `pyaudio`
- `pyserial`
- `matplotlib`
- `keyboard`

Install them with:

```bash
pip install numpy scipy pyaudio pyserial matplotlib keyboard
```

**Control tier (firmware):**
Follow the installation guidelines layed out in 

## Setup

1. **Wiring.** Connect the ReSpeaker array to the host PC by USB. Connect the STM32 to
   the host by USB/serial. Mount the X-NUCLEO-IHM01A1 on the Nucleo board and connect
   the pan and tilt motors.
2. **Firmware.** Open the `firmware/` project in STM32CubeIDE, build, and flash it to
   the board.
3. **Perception.** Install the Python dependencies (above) and set the serial port for
   your machine in `constants.py` / `robotics.py`.

## Usage

### Simulation (no hardware required)

Validate the DOA pipeline against a known ground truth. In `main.py`, choose a
trajectory and run:

```python
# main.py
mic.run(simulate=True, trajectory='arc')   # 'circle' | 'arc' | 'spiral'
```

```bash
python main.py
```

- **circle** — azimuth sweeps 360° at constant elevation (azimuth tracking).
- **arc** — fixed azimuth, elevation oscillates (elevation tracking).
- **spiral** — both angles vary together.

The run executes one full period of the trajectory, plots the live tracking error, and
prints summary statistics (mean, MAE, RMSE, std, max) for azimuth and elevation. Press
`w` to stop early. The trajectory generator is general — any custom `(phi(t), theta(t))`
path can be added.

### Live tracking

With the array connected and the firmware running:

```python
# main.py
mic.run(simulate=False)
```

The system selects the ReSpeaker device automatically, detects the dominant persistent
tone, estimates its direction, and streams the bearing to the microcontroller, which
steers the platform. Press `w` to stop.

### Binary tone detection

The detector flags a narrowband tone once it persists above the local adaptive
threshold for the required duration, and logs when each tone appears and vanishes. The
target tones used in this project are **400, 800, 1200, and 1600 Hz**.

## Serial protocol

The host streams each bearing to the microcontroller as a newline-terminated ASCII
line:

```
P:<phi>,T:<theta>\n
```

- `<phi>` — azimuth, radians
- `<theta>` — elevation, radians
- 115200 baud, 8-N-1

The format is human-readable and self-delimiting, so the receiver can resynchronise
cleanly after a corrupted line.

## Configuration

Key parameters live in `constants.py`:

- **Array geometry** — microphone radius / positions, used by the direction solve.
- **Audio** — sample rate and block (chunk) size.
- **Detection** — threshold factor and persistence length for the tone detector.
- **Smoothing** — the EMA factor for the bearing.

Motor tuning (PID gains, slew/derivative limits, gear ratios, heartbeat timeout) lives
in the firmware.

## Results

| Metric | Value |
|---|---|
| Azimuth RMSE | 1.84° |
| Elevation RMSE | 1.93° |
| Angular tolerance | within 5° |
| Noise rejection (front-end) | > 6 dB |
| Reaction time | < 1 s |
| Steady-state pointing error | sub-degree |

## Further work

- Use an array with more microphones and greater spacing to improve angular resolution.
- Extend the detector and tracker to multiple simultaneous sources.

## Authors

- **Mark Ruzal**
- **Roee Tabak**

Supervisor: **Arkady Rafalovich** · Tel Aviv University Project Lab.

## References

- Knapp, C. & Carter, G. *The generalized correlation method for estimation of time
  delay.* IEEE Trans. ASSP 24(4), 1976.
- Oppenheim, A. V. & Schafer, R. W. *Discrete-Time Signal Processing.* Pearson, 2009.
- STMicroelectronics. *STM32 HAL Driver Reference Manual.*

---

*License:* add a license file (e.g. MIT) and reference it here.
