import os

filepath = r'C:\Users\dell\Desktop\PCR ML\PCR_3step_PWM.ino'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Replace PCR PROTOCOL config
old_config = """// ============================================================
// PCR PROTOCOL
// ============================================================

#define TOTAL_CYCLES 40

#define INIT_DENAT_TARGET_C      95.0f
#define CYC_DENAT_TARGET_C       95.0f
#define ANNEAL_TARGET_C          60.0f
#define EXTENSION_TARGET_C       72.0f
#define FINAL_EXTENSION_TARGET_C 72.0f

// ============================================================
// HOLD TIME
// Change this ONE value to change all PCR hold times.
// Current protocol: 3 seconds.
// ============================================================

#define HOLD_TIME_S 30UL
#define LOG_INTERVAL_MS 1000UL

// ============================================================
// ML CALIBRATION VARIABLES (Model 3)
// ============================================================
float t_out_lag1 = NAN;
float t_out_lag2 = NAN;
float t_out_lag3 = NAN;
float estimated_t_in = NAN;"""

new_config = """// ============================================================
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
uint32_t lastMLUpdate = 0;"""

content = content.replace(old_config, new_config)


# 2. Update init variables
old_init = """float targetTemp = INIT_DENAT_TARGET_C;

uint32_t holdDuration = HOLD_TIME_S * 1000UL;"""

new_init = """float targetTemp = INITIAL_DENAT_TEMP;

uint32_t holdDuration = INITIAL_DENAT_TIME * 1000UL;"""
content = content.replace(old_init, new_init)


# 3. Add updateMLEstimate function before loop
old_setup = """// ============================================================
// SETUP
// ============================================================"""

new_setup = """// ============================================================
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
// ============================================================"""
content = content.replace(old_setup, new_setup)


# 4. Initialize ML array in setup
old_setup_end = """  ledPWM = 0;
  fanPWM = 0;

  Serial.println(F("===================================================="));"""

new_setup_end = """  ledPWM = 0;
  fanPWM = 0;

  for (int i = 0; i < 30; i++) {
    t_out_history[i] = NAN;
  }

  Serial.println(F("===================================================="));"""
content = content.replace(old_setup_end, new_setup_end)

# 5. Add updateMLEstimate to loop
old_loop_start = """  float s2 =
    tc_in.readCelsius();

  // ----------------------------------------------------------
  // S2 FAILURE
  // ----------------------------------------------------------"""

new_loop_start = """  float s2 =
    tc_in.readCelsius();

  // Update Continuous ML Estimate
  updateMLEstimate(now, s1);

  // ----------------------------------------------------------
  // S2 FAILURE
  // ----------------------------------------------------------"""
content = content.replace(old_loop_start, new_loop_start)


# 6. Replace target checks with estimated_t_in and ml_history_valid
old_ramp = """    // ========================================================
    // ANNEAL: 95 -> 60
    // Friend-style:
    // full cooling until target, then HOLD.
    // ========================================================

    if (currentPhase == CYCLE_ANNEAL) {

      if (controlMode == MAX_COOLING) {

        if (s2 <= targetTemp) {

          startHold(now);

        } else {

          maximumCooling();
        }
      }
    }

    // ========================================================
    // EXTENSION: 60 -> 72
    //
    // Initial 2 s full power is preserved.
    // Then continue full power until 70 C.
    // Then controlled approach to 72 C.
    // ========================================================

    else if (currentPhase == CYCLE_EXTENSION) {

      if (controlMode == MAX_HEATING) {

        // Full heating all the way to 72 C.
        // No PWM reduction before the target is reached.
        if (s2 >= targetTemp) {

          startHold(now);

        } else {

          maximumHeating();
        }
      }
    }

    // ========================================================
    // FINAL EXTENSION: current temp -> 72
    // Controlled heating.
    // ========================================================

    else if (currentPhase == FINAL_EXTENSION) {

      if (controlMode == MAX_HEATING) {

        // Full heating all the way to 72 C.
        // No PWM reduction before the target is reached.
        if (s2 >= targetTemp) {

          startHold(now);

        } else {

          maximumHeating();
        }
      }
    }

    // ========================================================
    // 95 C HEATING
    //
    // Friend-style:
    // FULL LED POWER until 95 C.
    // ========================================================

    else {

      if (controlMode == MAX_HEATING) {

        if (s2 >= DENAT_TARGET_C) {

          startHold(now);

        } else {

          maximumHeating();
        }
      }
    }"""

new_ramp = """    // ========================================================
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
    }"""
content = content.replace(old_ramp, new_ramp)


# 7. Replace filteredS2 in holdControl
old_hold = """  if (currentPhase == INIT_DENAT ||
      currentPhase == CYCLE_DENAT) {

    output = calculateHeatPID(
      now,
      95.0f,
      filteredS2,
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
      60.0f,
      filteredS2,
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
      72.0f,
      filteredS2,
      HOLD72_KP,
      HOLD72_KI,
      HOLD72_KD
    );
  }"""

new_hold = """  if (currentPhase == INIT_DENAT ||
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
  }"""
content = content.replace(old_hold, new_hold)


# 8. Replace nextPhase variables
old_nextphase = """  if (currentPhase == INIT_DENAT) {

    currentPhase = CYCLE_DENAT;

    targetTemp = CYC_DENAT_TARGET_C;

    holdDuration =
      HOLD_TIME_S * 1000UL;

    controlMode = MAX_HEATING;
  }

  // ----------------------------------------------------------
  // DENAT -> ANNEAL
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_DENAT) {

    currentPhase = CYCLE_ANNEAL;

    targetTemp = ANNEAL_TARGET_C;

    holdDuration =
      HOLD_TIME_S * 1000UL;

    controlMode = MAX_COOLING;
  }

  // ----------------------------------------------------------
  // ANNEAL -> EXTENSION
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_ANNEAL) {

    currentPhase = CYCLE_EXTENSION;

    targetTemp = EXTENSION_TARGET_C;

    holdDuration =
      HOLD_TIME_S * 1000UL;

    controlMode = MAX_HEATING;
  }

  // ----------------------------------------------------------
  // EXTENSION -> NEXT CYCLE / FINAL
  // ----------------------------------------------------------

  else if (currentPhase == CYCLE_EXTENSION) {

    if (currentCycle >= TOTAL_CYCLES) {

      currentPhase = FINAL_EXTENSION;

      targetTemp = FINAL_EXTENSION_TARGET_C;

      holdDuration =
        HOLD_TIME_S * 1000UL;

      // Final extension still reaches 72 using controlled
      // heating from the current temperature.
      controlMode = APPROACH_HEATING;

    } else {

      currentCycle++;

      currentPhase = CYCLE_DENAT;

      targetTemp = CYC_DENAT_TARGET_C;

      holdDuration =
        HOLD_TIME_S * 1000UL;

      controlMode = MAX_HEATING;
    }
  }"""

new_nextphase = """  if (currentPhase == INIT_DENAT) {

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
  }"""
content = content.replace(old_nextphase, new_nextphase)

# 9. Clean up logging block
old_logging = """    // ML CALIBRATION (Linear Model 3)
    if (!isnan(s1)) {
      if (!isnan(t_out_lag3)) {
        float dT_dt = s1 - t_out_lag1;
        
        estimated_t_in = 33.4161f 
                         + (0.7087f * s1) 
                         + (1.2542f * dT_dt) 
                         - (0.5456f * t_out_lag1) 
                         - (2.3649f * t_out_lag2) 
                         + (3.2085f * t_out_lag3) 
                         + (0.0060f * (float)currentCycle);
      }
      
      // Shift lags
      t_out_lag3 = t_out_lag2;
      t_out_lag2 = t_out_lag1;
      t_out_lag1 = s1;
    }

    Serial.print(F("Time:"));
    Serial.print(now / 1000.0, 1);

    Serial.print(F(", Cycle:"));
    Serial.print(currentCycle);

    Serial.print(F(", "));
    Serial.print(phaseName());

    Serial.print(F(", T_OUT:"));
    Serial.print(s1, 2);

    Serial.print(F(", T_IN_act:"));
    Serial.print(s2, 2);

    Serial.print(F(", T_IN_est:"));
    if (isnan(estimated_t_in)) {
      Serial.print(F("WAIT"));
    } else {
      Serial.print(estimated_t_in, 2);
    }
    
    Serial.print(F(", Error:"));
    if (isnan(estimated_t_in)) {
      Serial.print(F("WAIT"));
    } else {
      Serial.print(s2 - estimated_t_in, 2);
    }"""

new_logging = """    Serial.print(F("Time:"));
    Serial.print(now / 1000.0, 1);

    Serial.print(F(", Cycle:"));
    Serial.print(currentCycle);

    Serial.print(F(", "));
    Serial.print(phaseName());

    Serial.print(F(", T_OUT:"));
    Serial.print(s1, 2);

    Serial.print(F(", T_IN_act:"));
    Serial.print(s2, 2);

    Serial.print(F(", T_IN_est:"));
    if (ml_history_valid) {
      Serial.print(estimated_t_in, 2);
    } else {
      Serial.print(F("WAIT"));
    }
    
    Serial.print(F(", Error:"));
    if (ml_history_valid) {
      Serial.print(s2 - estimated_t_in, 2);
    } else {
      Serial.print(F("WAIT"));
    }"""
content = content.replace(old_logging, new_logging)


with open(filepath, 'w') as f:
    f.write(content)
print("Patched successfully!")
