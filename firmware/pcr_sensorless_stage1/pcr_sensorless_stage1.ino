// ==============================================================================
// 🧬 PCR-ML: STAGE 1 SENSORLESS THERMAL CONTROL FIRMWARE
// ==============================================================================
// Description:
//   Production firmware for an open-source IR-LED PCR thermocycler.
//   Predicts internal fluid temperature (T_IN) continuously at 10Hz using a
//   31-slot rolling circular buffer (~124 bytes RAM) fed exclusively from
//   the external block thermocouple (T_OUT).
//
// Stage 1 Operation:
//   - T_OUT + ML Model -> Complete PID Loop & State Machine Control
//   - T_IN (Internal Tube) -> Logging & In-Situ Error Validation Only
//
// Target: Arduino Nano (ATmega328P @ 16MHz)
// ==============================================================================

#include <SPI.h>
#include "Adafruit_MAX31855.h"
#include "hardware_pinout.h"
#include "ml_model_weights.h"

// --- HARDWARE SENSOR INTERFACES ---
Adafruit_MAX31855 tc_out(SPI_SCK_PIN, S1_CS_PIN, SPI_MISO_PIN); // S1: External Block / ML Input
Adafruit_MAX31855 tc_in (SPI_SCK_PIN, S2_CS_PIN, SPI_MISO_PIN); // S2: Internal Fluid / Validation Only

// --- PROTOCOL STATE MACHINE ---
enum PCRPhase {
  INIT_DENAT = 0,
  CYCLE_DENAT = 1,
  CYCLE_ANNEAL = 2,
  CYCLE_EXTENSION = 3,
  FINAL_EXTENSION = 4,
  FINISHED = 5
};

enum ControlMode {
  MAX_HEATING,
  MAX_COOLING,
  APPROACH_HEATING,
  APPROACH_COOLING,
  PID_HOLD
};

PCRPhase currentPhase = INIT_DENAT;
ControlMode controlMode = MAX_HEATING;
uint8_t currentCycle = 1;
bool isHolding = false;

// --- PID PARAMETERS ---
#define HOLD95_KP  35.0f
#define HOLD95_KI   1.0f
#define HOLD95_KD   1.5f

#define HOLD60_KP  30.0f
#define HOLD60_KI   0.8f
#define HOLD60_KD   1.0f

#define HOLD72_KP  28.0f
#define HOLD72_KI   0.8f
#define HOLD72_KD   1.0f

float targetTemp = DENAT_TARGET_C;
uint32_t holdDuration = INITIAL_DENAT_TIME_S * 1000UL;
uint32_t holdStartTime = 0;
float holdIntegral = 0.0f;
float holdLastError = 0.0f;
uint32_t holdLastPIDTime = 0;

int ledPWM = 0;
int fanPWM = 0;

// --- ML CONTINUOUS CIRCULAR BUFFER ---
#define ML_BUFFER_SIZE 31
float t_out_history[ML_BUFFER_SIZE];
int16_t t_out_head = -1;
uint8_t t_out_count = 0;
bool ml_history_valid = false;
float estimated_t_in = NAN;
float startup_bias = 0.0f;
float startup_s1 = 0.0f;
uint32_t lastMLUpdate = 0;
uint32_t lastLogTime = 0;

// ==============================================================================
// ACTUATOR DRIVER FUNCTIONS
// ==============================================================================

void allOff() {
  digitalWrite(LED_PIN, LOW);
  ledPWM = 0;
  digitalWrite(FAN_PIN, LOW);
  digitalWrite(FAN2_PIN, LOW);
  digitalWrite(FAN3_PIN, LOW);
  digitalWrite(FAN4_PIN, LOW);
  fanPWM = 0;
}

void maximumHeating() {
  digitalWrite(LED_PIN, HIGH);
  ledPWM = 255;
  digitalWrite(FAN_PIN, HIGH); // LED Heatsink fan runs during heating
  digitalWrite(FAN2_PIN, LOW);
  digitalWrite(FAN3_PIN, LOW);
  digitalWrite(FAN4_PIN, LOW);
  fanPWM = 0;
}

void maximumCooling() {
  digitalWrite(LED_PIN, LOW);
  ledPWM = 0;
  digitalWrite(FAN_PIN, LOW);
  digitalWrite(FAN2_PIN, HIGH);
  digitalWrite(FAN3_PIN, HIGH);
  digitalWrite(FAN4_PIN, HIGH);
  fanPWM = 255;
}

void setLED(int pwm) {
  pwm = constrain(pwm, 0, 255);
  ledPWM = pwm;
  analogWrite(LED_PIN, pwm);
  digitalWrite(FAN_PIN, (pwm > 0) ? HIGH : LOW);
}

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

// ==============================================================================
// PID CONTROLLER
// ==============================================================================

void resetHoldPID(uint32_t now) {
  holdIntegral = 0.0f;
  holdLastError = 0.0f;
  holdLastPIDTime = now;
}

void startHold(uint32_t now) {
  isHolding = true;
  holdStartTime = now;
  resetHoldPID(now);
}

float calculateHeatPID(uint32_t now, float sp, float pv, float kp, float ki, float kd) {
  if (now - holdLastPIDTime < PID_INTERVAL_MS) {
    return -1.0f;
  }
  float dt = (now - holdLastPIDTime) / 1000.0f;
  holdLastPIDTime = now;
  if (dt <= 0.0f || dt > 1.0f) dt = 0.1f;

  float error = sp - pv;
  holdIntegral += error * dt;
  holdIntegral = constrain(holdIntegral, -50.0f, 50.0f);

  float derivative = (error - holdLastError) / dt;
  holdLastError = error;

  float output = (kp * error) + (ki * holdIntegral) + (kd * derivative);
  return constrain(output, 0.0f, 255.0f);
}

void holdControl(uint32_t now) {
  float output = -1.0f;
  if (currentPhase == INIT_DENAT || currentPhase == CYCLE_DENAT) {
    output = calculateHeatPID(now, targetTemp, estimated_t_in, HOLD95_KP, HOLD95_KI, HOLD95_KD);
  } else if (currentPhase == CYCLE_ANNEAL) {
    output = calculateHeatPID(now, targetTemp, estimated_t_in, HOLD60_KP, HOLD60_KI, HOLD60_KD);
  } else if (currentPhase == CYCLE_EXTENSION || currentPhase == FINAL_EXTENSION) {
    output = calculateHeatPID(now, targetTemp, estimated_t_in, HOLD72_KP, HOLD72_KI, HOLD72_KD);
  }

  if (output >= 0.0f) {
    setLED((int)output);
    digitalWrite(FAN_PIN, (ledPWM > 0) ? HIGH : LOW);
    analogWrite(FAN2_PIN, 0);
    analogWrite(FAN3_PIN, 0);
    analogWrite(FAN4_PIN, 0);
    fanPWM = 0;
  }
}

// ==============================================================================
// STATE MACHINE TRANSITIONS
// ==============================================================================

void nextPhase() {
  isHolding = false;
  uint32_t now = millis();
  allOff();
  resetHoldPID(now);

  if (currentPhase == INIT_DENAT) {
    currentPhase = CYCLE_DENAT;
    targetTemp = DENAT_TARGET_C;
    holdDuration = DENAT_TIME_S * 1000UL;
    controlMode = MAX_HEATING;
  } else if (currentPhase == CYCLE_DENAT) {
    currentPhase = CYCLE_ANNEAL;
    targetTemp = ANNEAL_TARGET_C;
    holdDuration = ANNEAL_TIME_S * 1000UL;
    controlMode = MAX_COOLING;
  } else if (currentPhase == CYCLE_ANNEAL) {
    currentPhase = CYCLE_EXTENSION;
    targetTemp = EXTEND_TARGET_C;
    holdDuration = EXTEND_TIME_S * 1000UL;
    controlMode = MAX_HEATING;
  } else if (currentPhase == CYCLE_EXTENSION) {
    if (currentCycle >= TOTAL_CYCLES) {
      currentPhase = FINAL_EXTENSION;
      targetTemp = FINAL_EXTEND_TARGET_C;
      holdDuration = FINAL_TIME_S * 1000UL;
      controlMode = MAX_HEATING;
    } else {
      currentCycle++;
      currentPhase = CYCLE_DENAT;
      targetTemp = DENAT_TARGET_C;
      holdDuration = DENAT_TIME_S * 1000UL;
      controlMode = MAX_HEATING;
    }
  } else if (currentPhase == FINAL_EXTENSION) {
    currentPhase = FINISHED;
    allOff();
  }
}

// ==============================================================================
// ML ESTIMATION PIPELINE (10Hz Circular Buffer)
// ==============================================================================

void updateMLEstimate(uint32_t now, float s1) {
  if (isnan(s1)) return;

  if (now - lastMLUpdate >= ML_UPDATE_INTERVAL_MS) {
    lastMLUpdate = now;

    t_out_head = (int16_t)((t_out_head + 1) % ML_BUFFER_SIZE);
    t_out_history[t_out_head] = s1;

    if (t_out_count < ML_BUFFER_SIZE) {
      t_out_count++;
    }

    ml_history_valid = (t_out_count >= ML_BUFFER_SIZE);

    if (ml_history_valid) {
      int idx1 = (t_out_head - 10 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // 1.0s ago
      int idx2 = (t_out_head - 20 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // 2.0s ago
      int idx3 = (t_out_head - 30 + ML_BUFFER_SIZE) % ML_BUFFER_SIZE; // 3.0s ago

      float lag1 = t_out_history[idx1];
      float lag2 = t_out_history[idx2];
      float lag3 = t_out_history[idx3];
      float dT_dt = s1 - lag1;

      float raw_estimate = predictInternalTemperature(
        (uint8_t)currentPhase,
        s1,
        dT_dt,
        lag1,
        lag2,
        lag3,
        currentCycle
      );

      // Ambient startup gradient decay compensation
      static bool first_valid = true;
      if (first_valid) {
        startup_bias = raw_estimate - s1;
        if (startup_bias < 0.0f) startup_bias = 0.0f;
        startup_s1 = s1;
        first_valid = false;
      }

      estimated_t_in = raw_estimate;

      if (currentPhase == INIT_DENAT && startup_bias > 0.0f) {
        float heat_progress = (s1 - startup_s1) / 15.0f;
        if (heat_progress < 0.0f) heat_progress = 0.0f;
        if (heat_progress > 1.0f) heat_progress = 1.0f;
        float current_bias = startup_bias * (1.0f - heat_progress);
        estimated_t_in = raw_estimate - current_bias;
      }
    }
  }
}

// ==============================================================================
// ARDUINO SETUP & LOOP
// ==============================================================================

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  pinMode(FAN2_PIN, OUTPUT);
  pinMode(FAN3_PIN, OUTPUT);
  pinMode(FAN4_PIN, OUTPUT);

  allOff();

  for (int i = 0; i < ML_BUFFER_SIZE; i++) {
    t_out_history[i] = NAN;
  }

  Serial.println(F("===================================================="));
  Serial.println(F("🧬 PCR-ML: STAGE 1 SENSORLESS CONTROL INITIALIZED"));
  Serial.println(F("Time, Cycle, Phase, T_OUT, T_IN_act, T_IN_est, Error, LED_PWM, Status"));
  Serial.println(F("===================================================="));
}

void loop() {
  uint32_t now = millis();

  float s1 = tc_out.readCelsius(); // Control input
  float s2 = tc_in.readCelsius();  // Ground-truth validation sensor

  // Continuous ML update
  updateMLEstimate(now, s1);

  // Validation thermocouple check (Stage 1 logging only, does not halt control)
  if (isnan(s2)) {
    static uint32_t lastS2Warn = 0;
    if (now - lastS2Warn >= 2000UL) {
      lastS2Warn = now;
      Serial.println(F("WARN: T_IN disconnect. Control continues via ML estimate."));
    }
  }

  if (currentPhase == FINISHED) {
    allOff();
    return;
  }

  // --- RAMP CONTROL ---
  if (!isHolding) {
    if (currentPhase == CYCLE_ANNEAL) {
      if (controlMode == MAX_COOLING) {
        maximumCooling();
        if (ml_history_valid && estimated_t_in <= ANNEAL_TARGET_C) {
          startHold(now);
        }
      }
    } else {
      // Heating ramps (Denat or Extension)
      maximumHeating();
      if (ml_history_valid && estimated_t_in >= targetTemp) {
        startHold(now);
      }
    }
  } else {
    // --- HOLD CONTROL ---
    holdControl(now);
    if (now - holdStartTime >= holdDuration) {
      nextPhase();
    }
  }

  // --- TELEMETRY LOGGING (1Hz) ---
  if (now - lastLogTime >= LOG_INTERVAL_MS) {
    lastLogTime = now;
    float err = (isnan(s2) || isnan(estimated_t_in)) ? NAN : (estimated_t_in - s2);
    
    Serial.print(F("Time:")); Serial.print(now / 1000.0f, 1);
    Serial.print(F(", Cycle:")); Serial.print(currentCycle);
    Serial.print(F(", Phase:")); Serial.print((int)currentPhase);
    Serial.print(F(", T_OUT:")); Serial.print(s1, 2);
    Serial.print(F(", T_IN_act:")); Serial.print(s2, 2);
    Serial.print(F(", T_IN_est:")); Serial.print(estimated_t_in, 2);
    Serial.print(F(", Error:")); Serial.print(err, 2);
    Serial.print(F(", LED_PWM:")); Serial.print(ledPWM);
    Serial.print(F(", Status:")); Serial.println(isHolding ? F("HOLD") : F("RAMP"));
  }
}
