# Acoustic Tracking and Binary Detection

Real-time acoustic **Direction-of-Arrival (DOA)** tracking and **tone detection** on a
motorized pan–tilt platform. The system performs **binary (present / absent) detection** of narrowband
tones. A four-microphone array estimates the direction of the
dominant sound focusing around the detected frequency, and an STM32 microcontroller steers the platform to face it.

> **Project 25-1-1-3234** — Iby and Aladar Fleischman Faculty of Engineering, Tel Aviv University.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [How it works](#how-it-works)
- [Hardware](#hardware)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Serial protocol](#serial-protocol)
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
  (ping-pong) interrupt service routine, with a heartbeat timeout that stops the
  platform if the link is lost.
- **Built-in simulation** with circle / arc / spiral test trajectories and
  automatic error statistics (RMSE, MAE, max error).

## How it works

```
 Microphone array ──USB──▶  Host PC  (Python, perception tier)
                            ├─ tone detection  (statistical thresholding + hysteresis) ─▶ f*
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
| Microcontroller | STM32 Nucleo development board with an STM32F401RE microcontroller |
| Motor driver | X-NUCLEO-IHM01A1 stepper-driver expansion (L6474), one driver per axis |
| Actuators | Two stepper motors (pan and tilt) on a pan–tilt rig carrying the array |

> Pan and tilt use different gear ratios (≈ 3345 microsteps/rev on pan, 16576 on tilt),
> which the firmware accounts for in its step-to-angle conversion.


## Requirements

**Perception tier (Python 3):**

- `numpy`
- `scipy`
- `pyaudio`
- `pyserial`
- `matplotlib`

Install them with:

```bash
pip install numpy scipy pyaudio pyserial matplotlib
```

**Control tier (firmware):**

Follow the installation guidelines laid out in the [X-NUCLEO Installation Guide](docs/x-nucleo-installation-guide.pdf). After extracting the x-cube-spn1.zip onto your PC, replace the main.c, l6474_target_config.h and stm32f4xx_it.c files with the ones from this repo.  

## Setup

1. **Wiring.** Connect the ReSpeaker array to the host PC by USB. Connect the STM32 to
   the host by USB/serial. Mount the X-NUCLEO-IHM01A1 on the Nucleo board and connect
   the pan and tilt stepper motors (see [X-NUCLEO Installation Guide](docs/x-nucleo-installation-guide.pdf)). 
2. **Firmware.** Open the project in STM32CubeIDE, as instructed in the provided guide, build, and flash it to
   the board.
3. **Perception.** Install the Python dependencies (above) and set the serial port for
   your machine in `constants.py` / `robotics.py`.

## Usage

### Simulation

Validate the DOA pipeline against a known ground truth. In `main.py`, set SIMULATE = True and choose a
trajectory by setting SIM_TRAJECTORY to one of the following options:

- **circle** — azimuth sweeps 360° at constant elevation (azimuth tracking).
- **arc** — fixed azimuth, elevation oscillates (elevation tracking).
- **spiral** — both angles vary together.

The run executes one full period of the trajectory, plots the live tracking error, and
prints summary statistics (mean, MAE, RMSE, std, max) for azimuth and elevation. The trajectory generator is general — any custom `(phi(t), theta(t))`
path can be added.

### Live tracking

With the array connected and the firmware running, set SIMULATE = False and run the code.

The system selects the ReSpeaker device automatically, detects the dominant persistent
tone, estimates its direction, and streams the angles to the microcontroller, which
steers the platform. 

### Binary tone detection

The detector flags a narrowband tone once it persists above the local adaptive
threshold for the required duration, and logs when each tone appears and vanishes.

## Serial protocol

The host streams each bearing in radians to the microcontroller as a newline-terminated ASCII
line:

```
P:<phi>,T:<theta>\n
```
- 115200 baud

## Results

| Metric | Value |
|---|---|
| Azimuth RMSE | 1.84° |
| Elevation RMSE | 1.93° |
| Angular tolerance | within 5° |
| Noise rejection (front-end) | > 6 dB |
| Reaction time | < 1 s |
| Steady-state pointing error | within required range |

## Further work

- Use an array with more microphones and greater spacing to improve angular resolution.
- Use mic array with higher sampling rate.
- Integrating a Kalman filter.
- Stronger motors.
- Experiment with the Lawson norm minimization and whitening additional modes we have implemented in the perception code.

## Authors

- **Mark Ruzal**
- **Roee Tabak**

Supervisor: **Arkady Rafalovich** · Tel Aviv University Project Lab.

## References

- Knapp, C., & Carter, G. *The generalized correlation method for estimation of time delay.* IEEE Transactions on Acoustics, Speech, and Signal Processing 24.4 (1976): 320-327.
- Oppenheim, A. V., & Schafer, R. W. *Discrete-Time Signal Processing.* Pearson, 2009.
- STMicroelectronics. *STM32 HAL Driver Reference Manual.*
- *x-nucleo-ihm01a1 Quick Start Guide*, May 16, 2016.
- Greco, D. *Robust Blind Algorithm for DOA Estimation Using TDOA Consensus.* Acoustics 2025, 7, 52.
- Krishnaraj Varma, Takeshi Ikuma, & A. A. (Louis) Beex. *Robust TDE-Based DOA Estimation For Compact AUDIO Arrays.* Second IEEE Sensor Array and Multichannel Signal Processing Workshop (SAM2002), 4-6 August, 2002.
- J. O. Smith. "Quadratic Interpolation of Spectral Peaks," *Spectral Audio Signal Processing*. [Available online](https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html).
- G. Hunter. *Exponential Moving Average (EMA) Filters*. [Available online](https://blog.mbedded.ninja/programming/signal-processing/digital-filters/exponential-moving-average-ema-filter/).

---

*License:* add a license file (e.g. MIT) and reference it here.
