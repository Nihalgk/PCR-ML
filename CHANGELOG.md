# Changelog

All notable changes to the **PCR-ML** project are documented in this file.

---

## [v1.0.0] - Stage 1 Dual-Sensor In-Situ ML Validation & Package Release

### Added
- **Modular Python ML Package (`pcr_ml`)**:
  - Fast serial log parsing (`parser.py`) with support for varying legacy and new telemetry formats.
  - 100ms circular rolling buffer simulator (`features.py`) mimicking the exact firmware buffer.
  - Multi-model evaluation framework (`models.py`, `evaluation.py`) including Global Regression, Phase-Segmented Models (Denat/Anneal/Extend), Random Forest, and Gradient Boosting.
  - Automated Arduino C++ weight & header generator (`c_code_gen.py`).
  - Command-line interface (`cli.py`) for training, validation, and C++ header generation.
- **Production Firmware (`firmware/pcr_sensorless_stage1`)**:
  - 31-slot circular buffer implementation with zero memory leaks (124 bytes total RAM footprint on ATmega328P).
  - Dynamic startup bias estimation and smooth thermal gradient decay across initial ambient warmup.
  - Sensorless PID hold loop driven entirely by real-time $T_{in}$ estimates.
  - Rate-limited safety fail-safe logging for validation thermocouple disconnects.
- **Testing & Continuous Integration**:
  - Pytest automated test suite covering log parsing, feature alignment, model training, and C++/Python numerical parity.
  - GitHub Actions CI workflow supporting Python 3.10 through 3.13.
- **Documentation & Research Assets**:
  - Comprehensive publication-ready `README.md` with system architecture diagrams, BOM, and cross-validation benchmark tables.
  - Dataset manifest (`data/README.md`) detailing physical experimental run conditions.
  - Academic citation metadata (`CITATION.cff`).

---

## [v0.2.0] - Multi-Run Calibration & Stress Testing
- Conducted 40-cycle experimental runs across multiple hold protocols (3-second fast cycling and 30-second extended hold).
- Evaluated thermal soak and equilibrium delays between external block and internal fluid.
- Discovered 2s and 3s historical lag terms as primary predictors of thermal lag mass.

---

## [v0.1.0] - Initial Prototype & Hardware Setup
- Built open-source IR-LED thermocycler prototype with dual MAX31855 thermocouple sensors.
- Created Stage 0 baseline dual-sensor PID control firmware.
