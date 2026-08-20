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
// 30-sample rolling buffer for 3-second history at 100ms intervals
// Memory footprint: 30 floats * 4 bytes = 120 bytes RAM
float t_out_history[30];
uint8_t t_out_head = 0;
bool ml_history_valid = false;
float estimated_t_in = NAN;
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
// TEMPERATURE FILTER
// Used for PID holding.
// RAW S2 is used for target crossing.
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

      controlMode = APPROACH_HEATING;

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
    
    t_out_history[t_out_head] = s1;
    
    static bool wrapped = false;
    t_out_head++;
    if (t_out_head >= 30) {
      t_out_head = 0;
      wrapped = true;
    }
    
    ml_history_valid = wrapped;
    
    if (ml_history_valid) {
      int idx1 = (t_out_head + 30 - 10) % 30; // 1s lag
      int idx2 = (t_out_head + 30 - 20) % 30; // 2s lag
      int idx3 = t_out_head;                  // 3s lag
      
      float lag1 = t_out_history[idx1];
      float lag2 = t_out_history[idx2];
      float lag3 = t_out_history[idx3];
      
      float dT_dt = s1 - lag1;
      
      estimated_t_in = 33.4161f 
                       + (0.7087f * s1) 
                       + (1.2542f * dT_dt) 
                       - (0.5456f * lag1) 
                       - (2.3649f * lag2) 
                       + (3.2085f * lag3) 
                       + (0.0060f * (float)currentCycle);
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

  for (int i = 0; i < 30; i++) {
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
  // S2 FAILURE WARNING (NON-FATAL)
  // ----------------------------------------------------------

  if (isnan(s2)) {
    // Stage 1 Validation: Do NOT stop PCR if T_IN fails.
    // Just print a warning and let the ML control handle it.
    static uint32_t lastWarning = 0;
    if (now - lastWarning >= 2000UL) {
      Serial.println(F("WARNING: S2 thermocouple read failed (ignored for ML control)"));
      lastWarning = now;
    }
  }

  // ----------------------------------------------------------
  // UPDATE FILTER
  // ----------------------------------------------------------

  updateS2Filter(now, s2);

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

        // Use ML estimated T_IN. Require history to be valid before crossing.
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

    // Requested format: C:12  T_OUT:47.25  T_IN_act:71.50  T_IN_est:70.10  Err:1.40  PWM:52  EXTEND
    Serial.print(F("C:"));
    Serial.print(currentCycle);
    
    Serial.print(F("  T_OUT:"));
    Serial.print(s1, 2);
    
    Serial.print(F("  T_IN_act:"));
    if (isnan(s2)) Serial.print(F("NaN")); else Serial.print(s2, 2);
    
    Serial.print(F("  T_IN_est:"));
    if (ml_history_valid) Serial.print(estimated_t_in, 2); else Serial.print(F("WAIT"));
    
    Serial.print(F("  Err:"));
    if (ml_history_valid && !isnan(s2)) Serial.print(s2 - estimated_t_in, 2); else Serial.print(F("WAIT"));
    
    Serial.print(F("  PWM:"));
    Serial.print(ledPWM);
    
    Serial.print(F("  "));
    Serial.print(phaseName());
    
    Serial.print(F("  "));

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
