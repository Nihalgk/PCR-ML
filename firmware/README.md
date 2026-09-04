# ⚡ PCR-ML Firmware: Sensorless Stage 1 Controller

This directory contains the production Arduino C++ firmware for the **PCR-ML** thermocycler.

---

## 🎯 Architecture Overview

The firmware implements real-time machine learning inference directly inside the microcontroller main loop to eliminate the need for an internal reaction tube thermocouple (`T_IN`), predicting sample temperature purely from external aluminum block sensor telemetry (`T_OUT`).

```
[ External Sensor T_OUT (MAX31855) ]
                │
                ▼ (100ms Sampling / 10Hz)
┌──────────────────────────────────────────────┐
│  31-Slot Rolling Circular Buffer (124B RAM)  │
│  ├─ Current T_OUT                            │
│  ├─ First Derivative (dT/dt)                 │
│  ├─ Lag 1.0s (idx - 10)                      │
│  ├─ Lag 2.0s (idx - 20)                      │
│  └─ Lag 3.0s (idx - 30)                      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Phase-Segmented Linear Inference Engine     │
│  (Denaturation / Annealing / Extension)      │
│  Computation time: < 25 microseconds         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Continuous Temperature Estimate (T_IN_est)  │
│  ├─ Target Crossing Detection                │
│  └─ Closed-Loop PID Heater & Fan PWM Control │
└──────────────────────────────────────────────┘
```

---

## 📐 Microcontroller Resource Budget (ATmega328P @ 16MHz)

| Resource | Available | Firmware Usage | Percentage |
| :--- | :--- | :--- | :--- |
| **Flash Memory (Program)** | 30,720 bytes | ~11,240 bytes | 36.5% |
| **SRAM (Runtime Memory)** | 2,048 bytes | ~382 bytes (124B ML Buffer) | 18.6% |
| **Inference Latency** | 100,000 µs (10Hz period) | ~24.5 µs | < 0.03% CPU load |

---

## 🔌 Hardware Pinout & Wiring

| Pin | Function | Device / Connection | Logic / PWM Frequency |
| :--- | :--- | :--- | :--- |
| **D5** | LED Heater Output | High-Power IR-LED MOSFET Gate | 8-bit Timer0 (980 Hz PWM) |
| **D10** | Heatsink Fan | IR-LED Heatsink Fan Driver | Digital Follower (Active High) |
| **D3** | Cooling Fan 1 | Side Air Intake Fan MOSFET | 8-bit Timer2 (490 Hz PWM) |
| **D6** | Cooling Fan 2 | Chamber Exhaust Fan MOSFET | 8-bit Timer0 (980 Hz PWM) |
| **D9** | Cooling Fan 3 | Top Chamber Fan MOSFET | 8-bit Timer1 (490 Hz PWM) |
| **D13** | SPI SCK | MAX31855 Cold-Junction K-Type Sensor | Hardware SPI Clock |
| **D12** | SPI MISO | MAX31855 Data Out | Hardware SPI Data |
| **D8** | S1 CS | Foil Block Thermocouple Chip Select | Active Low |
| **D7** | S2 CS | Validation Tube Thermocouple Chip Select | Active Low |

---

## 🚀 Flashing & Compilation

### Option 1: Arduino IDE
1. Open `firmware/pcr_sensorless_stage1/pcr_sensorless_stage1.ino`.
2. Install the `Adafruit MAX31855` library via Library Manager.
3. Select Board: **Arduino Nano**, Processor: **ATmega328P** (or Old Bootloader).
4. Connect via USB and click **Upload**.

### Option 2: CLI (Arduino CLI)
```bash
arduino-cli compile --fqbn arduino:avr:nano:cpu=atmega328old firmware/pcr_sensorless_stage1/
arduino-cli upload -p COM3 --fqbn arduino:avr:nano:cpu=atmega328old firmware/pcr_sensorless_stage1/
```
