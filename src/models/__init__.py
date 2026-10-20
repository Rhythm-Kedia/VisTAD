from .cnn_1d import CNN1DStudent
from .bilstm import BiLSTMStudent
from .timesnet import TimesNetStudent

STUDENT_MODELS = {
    "cnn_1d": CNN1DStudent,
    "tcn": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "tcn_residual": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "bilstm": BiLSTMStudent,
    "timesnet": TimesNetStudent
}

def build_student(name: str, in_channels: int, **kwargs):
    """Instantiates a student model by name with consistent interface."""
    name = name.lower()
    if name not in STUDENT_MODELS:
        raise ValueError(f"Unknown student model '{name}'. Available: {list(STUDENT_MODELS.keys())}")
    return STUDENT_MODELS[name](in_channels=in_channels, **kwargs)

__all__ = ["CNN1DStudent", "BiLSTMStudent", "TimesNetStudent", "build_student", "STUDENT_MODELS"]
