# Contributing to PCR-ML

Thank you for your interest in contributing to **PCR-ML**! We welcome contributions that improve model accuracy, optimize edge firmware memory footprints, expand hardware support, or add experimental validation protocols.

---

## 🛠️ Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nihalgk/PCR-ML.git
   cd PCR-ML
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies and editable package:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Run the test suite:**
   ```bash
   pytest -v
   ```

---

## 🔬 Code Standards & Verification

- **Edge Constraints**: Any model proposed for deployment on microcontrollers (Arduino Nano / ATmega328P) must respect tight RAM constraints (< 200 bytes runtime RAM) and low compute overhead (no dynamic memory allocation).
- **C++/Python Parity**: If modifying feature extraction or model inference equations, you must run and pass `tests/test_cpp_parity.py` to ensure exact parity between Python simulation and Arduino firmware.
- **Data Integrity**: New experimental runs should be placed in `data/` along with protocol descriptions (sampling rate, cycle timings, sensor configurations).

---

## 🌿 Pull Request Process

1. Create a feature branch (`git checkout -b feature/thermal-soak-optimization`).
2. Implement your changes with accompanying unit tests in `tests/`.
3. Verify that all tests pass (`pytest -v`).
4. Commit with descriptive messages and submit a Pull Request against `main`.
