from .cnn_1d import CNN1DStudent
from .bilstm import BiLSTMStudent
from .timesnet import TimesNetStudent
from .stage_coupled_tcn import StageCoupledTCNStudent
from .gecco_coupled_tcn import GeccoSubsystemCoupledTCN
from .msl_coupled_tcn import MSLCoupledTCNStudent

STUDENT_MODELS = {
    "cnn_1d": CNN1DStudent,
    "tcn": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "tcn_residual": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "stage_coupled_tcn": lambda **kwargs: StageCoupledTCNStudent(**kwargs),
    "gecco_coupled_tcn": lambda **kwargs: GeccoSubsystemCoupledTCN(**kwargs),
    "msl_coupled_tcn": lambda **kwargs: MSLCoupledTCNStudent(**kwargs),
    "bilstm": BiLSTMStudent,
    "timesnet": TimesNetStudent
}

def build_student(name: str, in_channels: int = 51, **kwargs):
    """Instantiates a student model by name with consistent interface."""
    name = name.lower()
    if name not in STUDENT_MODELS:
        raise ValueError(f"Unknown student model '{name}'. Available: {list(STUDENT_MODELS.keys())}")
    if name == "stage_coupled_tcn":
        return StageCoupledTCNStudent(**kwargs)
    if name == "gecco_coupled_tcn":
        return GeccoSubsystemCoupledTCN(**kwargs)
    if name == "msl_coupled_tcn":
        return MSLCoupledTCNStudent(**kwargs)
    return STUDENT_MODELS[name](in_channels=in_channels, **kwargs)

__all__ = ["CNN1DStudent", "BiLSTMStudent", "TimesNetStudent", "StageCoupledTCNStudent", "GeccoSubsystemCoupledTCN", "MSLCoupledTCNStudent", "build_student", "STUDENT_MODELS"]
