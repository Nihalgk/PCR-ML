// ==============================================================================
// 🧬 PCR-ML: AUTO-GENERATED EMBEDDED THERMAL CALIBRATION WEIGHTS
// ==============================================================================
// Generated automatically by pcr_ml.c_code_gen at: 2026-09-04 12:40:53
// Target Platform: Arduino Nano (ATmega328P @ 16MHz)
// Memory footprint: ~124 bytes RAM (31-float circular buffer)
// ==============================================================================

#ifndef ML_MODEL_WEIGHTS_H
#define ML_MODEL_WEIGHTS_H

#include <Arduino.h>

// Model feature coefficients for Denaturation (95°C Target)
#define DENAT_INTERCEPT     72.010938f
#define DENAT_COEF_S1       -1.515973f
#define DENAT_COEF_DT_DT    -1.468915f
#define DENAT_COEF_LAG1     -0.047058f
#define DENAT_COEF_LAG2     0.522043f
#define DENAT_COEF_LAG3     1.324852f
#define DENAT_COEF_CYCLE    0.039710f

// Model feature coefficients for Annealing (60°C Target)
#define ANNEAL_INTERCEPT    51.237866f
#define ANNEAL_COEF_S1      -1.343978f
#define ANNEAL_COEF_DT_DT   -2.364223f
#define ANNEAL_COEF_LAG1    1.020245f
#define ANNEAL_COEF_LAG2    -2.008477f
#define ANNEAL_COEF_LAG3    2.743088f
#define ANNEAL_COEF_CYCLE   -0.000076f

// Model feature coefficients for Extension (72°C Target)
#define EXTEND_INTERCEPT    66.636269f
#define EXTEND_COEF_S1      -0.312420f
#define EXTEND_COEF_DT_DT   -0.432502f
#define EXTEND_COEF_LAG1    0.120083f
#define EXTEND_COEF_LAG2    0.218840f
#define EXTEND_COEF_LAG3    0.048506f
#define EXTEND_COEF_CYCLE   -0.020613f

/**
 * Inline real-time evaluation of internal PCR fluid temperature (T_IN).
 * Runs in < 25 microseconds on 16MHz ATmega328P without dynamic memory allocations.
 */
inline float predictInternalTemperature(
    uint8_t phaseId,
    float s1,
    float dT_dt,
    float lag1,
    float lag2,
    float lag3,
    uint8_t currentCycle
) {
    float raw_estimate = 0.0f;
    float cyc = (float)currentCycle;

    switch (phaseId) {
        case 0: // DENATURATION
        case 1:
            raw_estimate = DENAT_INTERCEPT
                         + (DENAT_COEF_S1 * s1)
                         + (DENAT_COEF_DT_DT * dT_dt)
                         + (DENAT_COEF_LAG1 * lag1)
                         + (DENAT_COEF_LAG2 * lag2)
                         + (DENAT_COEF_LAG3 * lag3)
                         + (DENAT_COEF_CYCLE * cyc);
            break;

        case 2: // ANNEALING
            raw_estimate = ANNEAL_INTERCEPT
                         + (ANNEAL_COEF_S1 * s1)
                         + (ANNEAL_COEF_DT_DT * dT_dt)
                         + (ANNEAL_COEF_LAG1 * lag1)
                         + (ANNEAL_COEF_LAG2 * lag2)
                         + (ANNEAL_COEF_LAG3 * lag3)
                         + (ANNEAL_COEF_CYCLE * cyc);
            break;

        case 3: // EXTENSION
        case 4:
            raw_estimate = EXTEND_INTERCEPT
                         + (EXTEND_COEF_S1 * s1)
                         + (EXTEND_COEF_DT_DT * dT_dt)
                         + (EXTEND_COEF_LAG1 * lag1)
                         + (EXTEND_COEF_LAG2 * lag2)
                         + (EXTEND_COEF_LAG3 * lag3)
                         + (EXTEND_COEF_CYCLE * cyc);
            break;

        default:
            raw_estimate = s1;
            break;
    }

    return raw_estimate;
}

#endif // ML_MODEL_WEIGHTS_H
