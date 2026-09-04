#include <SPI.h>
#include "Adafruit_MAX31855.h"

// ============================================================
// HARDWARE
// ============================================================

Adafruit_MAX31855 tc_out(13, 8, 12); // S1: Foil / monitoring only
Adafruit_MAX31855 tc_in (13, 7, 12); // S2: Tube / CONTROL sensor

#define LED_PIN 5

#define FAN_PIN 10   // D10: LED heatsink fan
#define FAN2_PIN 3
#define FAN3_PIN 6
#define FAN4_PIN 9

// ============================================================
// PCR SETTINGS
// ============================================================
#define TOTAL_CYCLES 40

#define INITIAL_DENAT_TEMP 95.0f
#define INITIAL_DENAT_TIME 60UL

#define DENAT_TEMP 95.0f
#define DENAT_TIME 30UL

#define ANNEAL_TEMP 60.0f
#define ANNEAL_TIME 30UL

#define EXTEND_TEMP 72.0f
#define EXTEND_TIME 30UL

#define FINAL_TEMP 72.0f
#define FINAL_TIME 60UL

#define LOG_INTERVAL_MS 1000UL

// ============================================================
// ML CALIBRATION VARIABLES (Model 3)
// ============================================================
// Rolling buffer sampled every 100ms. To recover a sample from
// EXACTLY 3.0s ago (30 steps back) as well as the current sample,
// the buffer must hold 31 slots (0..30 steps back), not 30.
// Memory footprint: 31 floats * 4 bytes = 124 bytes RAM
#define ML_BUFFER_SIZE 31
float t_out_history[ML_BUFFER_SIZE];
// t_out_head always points at the slot holding the MOST RECENT
// sample (not the next-empty slot). Starts at -1 so the first
// write lands at index 0.
int16_t t_out_head = -1;
uint8_t t_out_count = 0; // number of real samples written, capped at ML_BUFFER_SIZE
bool ml_history_valid = false;
float estimated_t_in = NAN;
float startup_bias = 0.0f;
float startup_s1 = 0.0f;
uint32_t lastMLUpdate = 0;

// ============================================================
// RAMP SETTINGS
// ============================================================

// 95 C:full heater until target.
#define DENAT_TARGET_C            95.0f

// 60 C cooling: full cooling until target.
#define ANNEAL_TARGET_C_FULL      60.0f

// 60 -> 72 C:
// Full power until 71 C, then controlled approach to 72 C.
#define EXTENSION_FULL_TO_C      71.0f

// ============================================================
// TEMPERATURE FILTER (S2 / T_IN)
// AUDIT NOTE: filteredS2 is computed below (updateS2Filter) but is
// NOT read anywhere in the current control path -- holdControl()
// drives PID entirely off estimated_t_in, and target crossing checks
// use estimated_t_in as well. filteredS2 is inert/vestigial in Stage
// 1. Kept as-is (harmless) rather than removed, since T_IN is
// validation-only and this doesn't affect control; flagging here so
// nobody mistakes the old comment below for current behavior.
// ============================================================

#define S2_FILTER_ALPHA          0.70f

// ============================================================
// PID UPDATE
// ============================================================

#define PID_INTERVAL_MS          100UL

// ============================================================
// HOLD PID PARAMETERS
// Same control concept as friend's setup:
// full power during ramp, PID only during HOLD.
// ============================================================

// 95 C HOLD
#define HOLD95_KP                 35.0f
#define HOLD95_KI                  1.0f
#define HOLD95_KD                  1.5f

// 60 C HOLD
// LED adds heat while D10 remains always ON.
#define HOLD60_KP                 30.0f
#define HOLD60_KI                  0.8f
#define HOLD60_KD                  1.0f

// 72 C HOLD
#define HOLD72_KP                 28.0f
#define HOLD72_KI                  0.8f
#define HOLD72_KD                  1.0f

// ============================================================

enum Phase {
  INIT_DENAT,
  CYCLE_DENAT,
  CYCLE_ANNEAL,
  CYCLE_EXTENSION,
  FINAL_EXTENSION,
  FINISHED
};

Phase currentPhase = INIT_DENAT;

int currentCycle = 1;

float targetTemp = INITIAL_DENAT_TEMP;

uint32_t holdDuration = INITIAL_DENAT_TIME * 1000UL;
uint32_t holdStartTime = 0;

bool isHolding = false;

// ============================================================
// CONTROL MODE
// ============================================================

enum ControlMode {
  MAX_HEATING,
  APPROACH_HEATING,
  MAX_COOLING,
  APPROACH_COOLING
};

ControlMode controlMode = MAX_HEATING;

// ============================================================
// S2 FILTER
// ============================================================

float filteredS2 = NAN;
uint32_t lastS2FilterTime = 0;

// ============================================================
// ============================================================
// PID VARIABLES
// ============================================================

float holdIntegral = 0.0f;
float holdLastError = 0.0f;
uint32_t holdLastPIDTime = 0;

// ============================================================
// OUTPUT VARIABLES
// ============================================================

int ledPWM = 0;
int fanPWM = 0;

// ============================================================
// PHASE NAME
// ============================================================

const char* phaseName() {

  if (currentPhase == INIT_DENAT)
    return "INIT";

  if (currentPhase == CYCLE_DENAT)
    return "DENAT";

  if (currentPhase == CYCLE_ANNEAL)
    return "ANNEAL";

  if (currentPhase == CYCLE_EXTENSION)
    return "EXTEND";

  if (currentPhase == FINAL_EXTENSION)
    return "FINAL";

  return "DONE";
}

// ============================================================
// ALL OUTPUTS OFF
// ============================================================

void allOff() {

  digitalWrite(LED_PIN, LOW);
  ledPWM = 0;

  // D10 follows LED state.
  digitalWrite(FAN_PIN, LOW);

  digitalWrite(FAN2_PIN, LOW);
  digitalWrite(FAN3_PIN, LOW);
  digitalWrite(FAN4_PIN, LOW);

  fanPWM = 0;
}

// ============================================================
// FULL HEATING
// ============================================================

void maximumHeating() {

  digitalWrite(LED_PIN, HIGH);
  ledPWM = 255;

  // D10 follows LED state.
  digitalWrite(FAN_PIN, HIGH);

  // No additional cooling while heating.
  digitalWrite(FAN2_PIN, LOW);
  digitalWrite(FAN3_PIN, LOW);
  digitalWrite(FAN4_PIN, LOW);

  fanPWM = 0;
}

// ============================================================
// FULL COOLING
// LED off; all cooling fans ON.
// ============================================================

void maximumCooling() {

  digitalWrite(LED_PIN, LOW);
  ledPWM = 0;

  digitalWrite(FAN_PIN, LOW);
  digitalWrite(FAN2_PIN, HIGH);
  digitalWrite(FAN3_PIN, HIGH);
  digitalWrite(FAN4_PIN, HIGH);

  fanPWM = 255;
}

// ============================================================
// SET LED PWM
// ============================================================

void setLED(int pwm) {

  pwm = constrain(pwm, 0, 255);

  ledPWM = pwm;

  analogWrite(LED_PIN, pwm);

  // D10 follows LED state.
  digitalWrite(FAN_PIN, (pwm > 0) ? HIGH : LOW);
}

// ============================================================
// SET FAN PWM
// PWM is applied to D3, D6 and D9.
// ============================================================

void setFan(int pwm) {

  pwm = constrain(pwm, 0, 255);

  fanPWM = pwm;

  digitalWrite(FAN_PIN, (ledPWM > 0) ? HIGH : LOW);

  analogWrite(FAN2_PIN, pwm);
  analogWrite(FAN3_PIN, pwm);
  analogWrite(FAN4_PIN, pwm);

  if (pwm > 0) {
    digitalWrite(LED_PIN, LOW);
    ledPWM = 0;
  }
}

// ============================================================
// UPDATE S2 FILTER
// ============================================================

void updateS2Filter(uint32_t now, float rawS2) {

  if (isnan(rawS2))
    return;

  if (isnan(filteredS2)) {

    filteredS2 = rawS2;

    lastS2FilterTime = now;

    return;
  }

  if (now - lastS2FilterTime >= 50UL) {

    lastS2FilterTime = now;

    filteredS2 =
      filteredS2 +
      S2_FILTER_ALPHA *
      (rawS2 - filteredS2);
  }
}

// ============================================================
// RESET HOLD PID
// ============================================================

void resetHoldPID(uint32_t now) {

  holdIntegral = 0.0f;
  holdLastError = 0.0f;
  holdLastPIDTime = now;
}

// ============================================================
// START HOLD
// ============================================================

void startHold(uint32_t now) {

  isHolding = true;

  holdStartTime = now;

  resetHoldPID(now);
}

// ============================================================
// GENERIC PID FOR HEATER HOLD
// Friend-style logic:
// PID output is directly the LED PWM.
// ============================================================

float calculateHeatPID(
  uint32_t now,
  float target,
  float temperature,
  float kp,
  float ki,
  float kd
) {

  if (now - holdLastPIDTime < PID_INTERVAL_MS)
    return -1.0f;

  float dt =
    (now - holdLastPIDTime) / 1000.0f;

  if (dt <= 0.0f)
    dt = 0.1f;

  holdLastPIDTime = now;

  float error =
    target - temperature;

  holdIntegral += ki * error * dt;

  // Anti-windup in output units.
  if (holdIntegral > 255.0f)
    holdIntegral = 255.0f;

  if (holdIntegral < 0.0f)
    holdIntegral = 0.0f;

  float derivative =
    (error - holdLastError) / dt;

  holdLastError = error;

  float output =
    (kp * error) +
    holdIntegral +
    (kd * derivative);

  if (output < 0.0f)
    output = 0.0f;

  if (output > 255.0f)
    output = 255.0f;

  return output;
}

// ============================================================
// HOLD CONTROL
// ============================================================

void holdControl(uint32_t now) {

  float output = -1.0f;

  // ----------------------------------------------------------
  // 95 C HOLD
  // ----------------------------------------------------------

  if (currentPhase == INIT_DENAT ||
      currentPhase == CYCLE_DENAT) {

    output = calculateHeatPID(
      now,
      targetTemp,
      estimated_t_in, // ML ESTIMATE DRIVES PID
      HOLD95_KP,
      HOLD95_KI,
      HOLD95_KD
    );
  }

  // ----------------------------------------------------------
  // 60 C HOLD
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_ANNEAL) {

    output = calculateHeatPID(
      now,
      targetTemp,
      estimated_t_in, // ML ESTIMATE DRIVES PID
      HOLD60_KP,
      HOLD60_KI,
      HOLD60_KD
    );
  }

  // ----------------------------------------------------------
  // 72 C HOLD
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_EXTENSION ||
           currentPhase == FINAL_EXTENSION) {

    output = calculateHeatPID(
      now,
      targetTemp,
      estimated_t_in, // ML ESTIMATE DRIVES PID
      HOLD72_KP,
      HOLD72_KI,
      HOLD72_KD
    );
  }

  if (output >= 0.0f) {

    setLED((int)output);

    // D10 follows LED state.
    // Other fans OFF during heating holds.
    digitalWrite(FAN_PIN, (ledPWM > 0) ? HIGH : LOW);
    analogWrite(FAN2_PIN, 0);
    analogWrite(FAN3_PIN, 0);
    analogWrite(FAN4_PIN, 0);
    fanPWM = 0;
  }
}

// ============================================================
// NEXT PHASE
// ============================================================

void nextPhase() {

  isHolding = false;

  uint32_t now = millis();

  // Safety transition.
  allOff();

  resetHoldPID(now);

  // ----------------------------------------------------------
  // INIT -> DENAT
  // ----------------------------------------------------------

  if (currentPhase == INIT_DENAT) {

    currentPhase = CYCLE_DENAT;

    targetTemp = DENAT_TEMP;

    holdDuration = DENAT_TIME * 1000UL;

    controlMode = MAX_HEATING;
  }

  // ----------------------------------------------------------
  // DENAT -> ANNEAL
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_DENAT) {

    currentPhase = CYCLE_ANNEAL;

    targetTemp = ANNEAL_TEMP;

    holdDuration = ANNEAL_TIME * 1000UL;

    controlMode = MAX_COOLING;
  }

  // ----------------------------------------------------------
  // ANNEAL -> EXTENSION
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_ANNEAL) {

    currentPhase = CYCLE_EXTENSION;

    targetTemp = EXTEND_TEMP;

    holdDuration = EXTEND_TIME * 1000UL;

    controlMode = MAX_HEATING;
  }

  // ----------------------------------------------------------
  // EXTENSION -> NEXT CYCLE / FINAL
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_EXTENSION) {

    if (currentCycle >= TOTAL_CYCLES) {

      currentPhase = FINAL_EXTENSION;

      targetTemp = FINAL_TEMP;

      holdDuration = FINAL_TIME * 1000UL;

      // Bug fix: FINAL_EXTENSION's ramp-control branch checks for
      // MAX_HEATING (same as CYCLE_EXTENSION). APPROACH_HEATING has no
      // handling anywhere in the ramp control block, so leaving this as
      // APPROACH_HEATING silently stalled the final phase forever after
      // allOff() -- it never called maximumHeating(), never re-checked
      // estimated_t_in against targetTemp, and so never reached FINAL_TEMP
      // or started the final hold.
      controlMode = MAX_HEATING;

    } else {

      currentCycle++;

      currentPhase = CYCLE_DENAT;

      targetTemp = DENAT_TEMP;

      holdDuration = DENAT_TIME * 1000UL;

      controlMode = MAX_HEATING;
    }
  }

  // ----------------------------------------------------------
  // FINAL -> FINISHED
  // ----------------------------------------------------------

  else if (currentPhase == FINAL_EXTENSION) {

    currentPhase = FINISHED;
  }
}

// ============================================================
// ML CALIBRATION LOGIC
// ============================================================

void updateMLEstimate(uint32_t now, float s1) {
  if (isnan(s1)) return;

  if (now - lastMLUpdate >= 100UL) {
    lastMLUpdate = now;

    // Advance head to the new slot, then write the current sample there.
    // Head always points at the newest sample, so "n samples ago" is
    // simply (head - n) mod ML_BUFFER_SIZE -- no off-by-one ambiguity.
    t_out_head = (int16_t)((t_out_head + 1) % ML_BUFFER_SIZE);
    t_out_history[t_out_head] = s1;

    if (t_out_count < ML_BUFFER_SIZE) {
      t_out_count++;
    }

    // Only trust lag values once the buffer is completely full of real
    // samples -- i.e. we truly have a full 3.0s window, not NaN filler.
    ml_history_valid = (t_out_count >= ML_BUFFER_SIZE);

    if (ml_history_valid) {
      int idx1 = (t_out_head - 10 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // exactly 1.0s ago
      int idx2 = (t_out_head - 20 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // exactly 2.0s ago
      int idx3 = (t_out_head - 30 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // exactly 3.0s ago

      float lag1 = t_out_history[idx1];
      float lag2 = t_out_history[idx2];
      float lag3 = t_out_history[idx3];

      float dT_dt = s1 - lag1; // change over exactly the last 1.0s

      float raw_estimate = 0.0f;
      
      if (currentPhase == INIT_DENAT || currentPhase == CYCLE_DENAT) {
        raw_estimate = 35.4544f
                     + (0.6498f * s1)
                     + (0.3914f * dT_dt)
                     + (0.2584f * lag1)
                     - (0.0054f * lag2)
                     + (0.0798f * lag3)
                     + (0.0133f * (float)currentCycle);
      } else if (currentPhase == CYCLE_ANNEAL) {
        raw_estimate = 20.9982f
                     + (0.4435f * s1)
                     + (0.0451f * dT_dt)
                     + (0.3984f * lag1)
                     + (0.2361f * lag2)
                     + (0.1821f * lag3)
                     - (0.0172f * (float)currentCycle);
      } else if (currentPhase == CYCLE_EXTENSION || currentPhase == FINAL_EXTENSION) {
        raw_estimate = 53.7299f
                     + (0.1665f * s1)
                     + (0.2168f * dT_dt)
                     - (0.0503f * lag1)
                     + (0.0990f * lag2)
                     + (0.2051f * lag3)
                     - (0.0022f * (float)currentCycle);
      } else {
        // Fallback
        raw_estimate = s1; 
      }

      // --- STARTUP DATA-DRIVEN FIX ---
      // Analysis of the 5-run dataset shows a ~33C thermal gradient between 
      // T_IN and T_OUT during steady-state holds (e.g. at 60C ANNEAL, T_OUT is ~27C).
      // The linear model correctly learned this gradient to maintain accuracy during PCR.
      // Therefore, at true room temperature (25C), the model calculates: 
      // 25C + 33C intercept = 58C.
      // To fix the startup display jump without hardcoding arbitrary bounds or ruining 
      // the PCR holds, we capture the exact model bias on the very first valid reading 
      // (when the machine is at uniform ambient temp) and smoothly decay this bias to 0 
      // as T_OUT rises by its first 15C (which is the physical time required for the 
      // LED heater to establish the actual thermal gradient across the aluminum foil).
      
      static bool first_valid = true;
      
      if (first_valid) {
        startup_bias = raw_estimate - s1; // Typically ~33.0C
        if (startup_bias < 0.0f) startup_bias = 0.0f; 
        startup_s1 = s1;
        first_valid = false;
      }
      
      estimated_t_in = raw_estimate;
      
      if (currentPhase == INIT_DENAT && startup_bias > 0.0f) {
        // Smoothly decay the bias as the aluminum block heats up
        float heat_progress = (s1 - startup_s1) / 15.0f; 
        if (heat_progress < 0.0f) heat_progress = 0.0f;
        if (heat_progress > 1.0f) heat_progress = 1.0f;
        
        float current_bias = startup_bias * (1.0f - heat_progress);
        estimated_t_in = raw_estimate - current_bias;
      }
    }
  }
}

// ============================================================
// SETUP
// ============================================================

void setup() {

  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);

  pinMode(FAN_PIN, OUTPUT);
  pinMode(FAN2_PIN, OUTPUT);
  pinMode(FAN3_PIN, OUTPUT);
  pinMode(FAN4_PIN, OUTPUT);

  // D10 follows LED state and starts OFF with the LED.
  digitalWrite(FAN_PIN, LOW);

  // Start with heater and other fans OFF.
  digitalWrite(LED_PIN, LOW);
  digitalWrite(FAN2_PIN, LOW);
  digitalWrite(FAN3_PIN, LOW);
  digitalWrite(FAN4_PIN, LOW);

  ledPWM = 0;
  fanPWM = 0;

  for (int i = 0; i < ML_BUFFER_SIZE; i++) {
    t_out_history[i] = NAN;
  }

  Serial.println(F("===================================================="));
  Serial.println(F("PCR: ML STAGE 1 VALIDATION"));
  Serial.println(F("Time, Cyc, Phase, T_OUT, T_IN_act, T_IN_est, Error, LED PWM, Status"));
  Serial.println(F("===================================================="));
}

// ============================================================
// LOOP
// ============================================================


void loop() {

  uint32_t now = millis();

  float s1 =
    tc_out.readCelsius();

  float s2 =
    tc_in.readCelsius();

  // Update Continuous ML Estimate
  updateMLEstimate(now, s1);

  // ----------------------------------------------------------
  // S2 (T_IN) FAILURE
  // T_IN is VALIDATION/LOGGING ONLY in Stage 1. A failed read must
  // NOT stop, cool, or otherwise touch control -- control keeps
  // running entirely off estimated_t_in. We only log a rate-limited
  // warning and skip the S2 filter update (which is itself unused
  // for control -- see note above S2_FILTER_ALPHA).
  // ----------------------------------------------------------

  bool s2_failed = isnan(s2);

  if (s2_failed) {

    static uint32_t lastS2ErrorLog = 0;

    if (now - lastS2ErrorLog >= 1000UL) {

      lastS2ErrorLog = now;

      Serial.println(
        F("WARNING: T_IN (S2) read failed -- validation sensor only, PCR control continues on ML estimate")
      );
    }

  } else {

    // ----------------------------------------------------------
    // UPDATE FILTER
    // ----------------------------------------------------------

    updateS2Filter(now, s2);
  }

  // ----------------------------------------------------------
  // FINISHED
  // ----------------------------------------------------------

  if (currentPhase == FINISHED) {

    // Finished = LED OFF and all fans OFF.
    digitalWrite(LED_PIN, LOW);
    ledPWM = 0;

    digitalWrite(FAN_PIN, LOW);

    digitalWrite(FAN2_PIN, LOW);
    digitalWrite(FAN3_PIN, LOW);
    digitalWrite(FAN4_PIN, LOW);

    fanPWM = 0;

    return;
  }

  // ==========================================================
  // RAMP CONTROL
  // ==========================================================

  if (!isHolding) {

    // ========================================================
    // ANNEAL: 95 -> 60
    // ========================================================

    if (currentPhase == CYCLE_ANNEAL) {

      if (controlMode == MAX_COOLING) {

        if (ml_history_valid && estimated_t_in <= targetTemp) {
          startHold(now);
        } else {
          maximumCooling();
        }
      }
    }

    // ========================================================
    // EXTENSION: 60 -> 72
    // ========================================================

    else if (currentPhase == CYCLE_EXTENSION) {

      if (controlMode == MAX_HEATING) {

        if (ml_history_valid && estimated_t_in >= targetTemp) {
          startHold(now);
        } else {
          maximumHeating();
        }
      }
    }

    // ========================================================
    // FINAL EXTENSION
    // ========================================================

    else if (currentPhase == FINAL_EXTENSION) {

      if (controlMode == MAX_HEATING) {

        if (ml_history_valid && estimated_t_in >= targetTemp) {
          startHold(now);
        } else {
          maximumHeating();
        }
      }
    }

    // ========================================================
    // 95 C HEATING
    // ========================================================

    else {

      if (controlMode == MAX_HEATING) {

        if (ml_history_valid && estimated_t_in >= targetTemp) {
          startHold(now);
        } else {
          maximumHeating();
        }
      }
    }
  }

  // ==========================================================
  // HOLD
  // ==========================================================

  if (isHolding) {

    holdControl(now);

    // EXACT programmed hold time.
    if (now - holdStartTime >= holdDuration) {

      nextPhase();
    }
  }

  // ==========================================================
  // LOGGING
  // Same basic format as your current output,
  // with LED PWM added.
  // ==========================================================

  static uint32_t tLog = 0;

  if (now - tLog >= LOG_INTERVAL_MS) {

    tLog = now;

    Serial.print(F("Time:"));
    Serial.print(now / 1000.0, 1);

    Serial.print(F(", Cycle:"));
    Serial.print(currentCycle);

    Serial.print(F(", "));
    Serial.print(phaseName());

    Serial.print(F(", T_OUT:"));
    Serial.print(s1, 2);

    Serial.print(F(", T_IN_act:"));
    if (s2_failed) {
      Serial.print(F("NaN"));
    } else {
      Serial.print(s2, 2);
    }

    Serial.print(F(", T_IN_est:"));
    if (ml_history_valid) {
      Serial.print(estimated_t_in, 2);
    } else {
      Serial.print(F("WAIT"));
    }

    Serial.print(F(", Error:"));
    if (ml_history_valid && !s2_failed) {
      Serial.print(s2 - estimated_t_in, 2);
    } else {
      Serial.print(F("WAIT"));
    }

    Serial.print(F(", LED PWM:"));
    Serial.print(ledPWM);

    Serial.print(F(", "));

    if (isHolding) {

      uint32_t elapsed =
        now - holdStartTime;

      uint32_t secLeft = 0;

      if (elapsed < holdDuration) {

        secLeft =
          (holdDuration - elapsed) / 1000UL;
      }

      if (currentPhase == INIT_DENAT ||
          currentPhase == CYCLE_DENAT) {

        Serial.print(F("HOLD95 ("));

      } else if (currentPhase == CYCLE_ANNEAL) {

        Serial.print(F("HOLD60 ("));

      } else {

        Serial.print(F("HOLD72 ("));
      }

      Serial.print(secLeft);

      Serial.println(F("s left)"));
    }

    else if (controlMode == MAX_HEATING) {

      if (currentPhase == CYCLE_EXTENSION) {

        Serial.println(
          F("HEATING_FULL")
        );

      } else {

        Serial.println(
          F("HEATING_FULL")
        );
      }
    }

    else if (controlMode == APPROACH_HEATING) {

      Serial.println(
        F("HEATING")
      );
    }

    else if (controlMode == MAX_COOLING) {

      Serial.println(
        F("COOLING_FULL")
      );
    }

    else if (controlMode == APPROACH_COOLING) {

      Serial.println(
        F("COOLING")
      );
    }
  }
}
