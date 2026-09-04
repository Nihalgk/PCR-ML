import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from .parser import parse_pcr_file, load_all_runs
from .features import engineer_features, extract_phase_supercategories
from .models import PCRLinearModel, PhaseSpecificPCRModel, PCRRandomForestModel
from .evaluation import evaluate_leave_one_out, evaluate_model_pipeline
from .c_code_gen import generate_arduino_header

__version__ = "1.0.0"
__author__ = "Nihal G K"

__all__ = [
    "parse_pcr_file",
    "load_all_runs",
    "engineer_features",
    "extract_phase_supercategories",
    "PCRLinearModel",
    "PhaseSpecificPCRModel",
    "PCRRandomForestModel",
    "evaluate_leave_one_out",
    "evaluate_model_pipeline",
    "generate_arduino_header",
]
