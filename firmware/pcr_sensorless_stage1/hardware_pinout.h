// ==============================================================================
// 🧬 PCR-ML: HARDWARE PINOUT AND CONFIGURATION DEFINITIONS
// ==============================================================================
// Target Microcontroller: Arduino Nano (ATmega328P, 16MHz, 5V Logic)
// ==============================================================================

#ifndef HARDWARE_PINOUT_H
#define HARDWARE_PINOUT_H

#include <Arduino.h>

// --- ACTUATOR OUTPUT PINS (PWM & DIGITAL) ---
#define LED_PIN          5   // D5: High-Power IR-LED Heater (Timer0 980Hz PWM)
#define FAN_PIN         10   // D10: LED Heatsink Protection Fan (Digital Follower)
#define FAN2_PIN         3   // D3: Chamber Cooling Fan 1 (Timer2 490Hz PWM)
#define FAN3_PIN         6   // D6: Chamber Cooling Fan 2 (Timer0 980Hz PWM)
#define FAN4_PIN         9   // D9: Chamber Cooling Fan 3 (Timer1 490Hz PWM)

// --- THERMOCOUPLE SENSOR SPI PINS (MAX31855 Cold-Junction K-Type Amplifiers) ---
#define SPI_SCK_PIN     13   // D13: SPI Shared Clock
#define SPI_MISO_PIN    12   // D12: SPI Shared MISO (Data Out)
#define S1_CS_PIN        8   // D8: S1 Chip Select (T_OUT - Aluminum Block / Control Input)
#define S2_CS_PIN        7   // D7: S2 Chip Select (T_IN - Reaction Tube / Validation Sensor)

// --- PROTOCOL THERMAL TARGETS (°C) ---
#define DENAT_TARGET_C           95.0f
#define ANNEAL_TARGET_C          60.0f
#define EXTEND_TARGET_C          72.0f
#define FINAL_EXTEND_TARGET_C    72.0f

// --- TIMINGS (Seconds) ---
#define TOTAL_CYCLES             40
#define INITIAL_DENAT_TIME_S     60UL
#define DENAT_TIME_S             30UL
#define ANNEAL_TIME_S            30UL
#define EXTEND_TIME_S            30UL
#define FINAL_TIME_S             60UL

// --- TELEMETRY & CONTROL INTERVALS (Milliseconds) ---
#define ML_UPDATE_INTERVAL_MS   100UL  // ML buffer sampling rate (10Hz)
#define PID_INTERVAL_MS         100UL  // Closed-loop PID frequency (10Hz)
#define LOG_INTERVAL_MS        1000UL  // Serial telemetry logging rate (1Hz)

#endif // HARDWARE_PINOUT_H
