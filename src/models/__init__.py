from .cnn_1d import CNN1DStudent
from .bilstm import BiLSTMStudent
from .timesnet import TimesNetStudent
from .stage_coupled_tcn import StageCoupledTCNStudent

try:
    from .smd_coupled_tcn import SMDSubsystemCoupledTCN
except ImportError:
    SMDSubsystemCoupledTCN = None

try:
    from .gecco_coupled_tcn import GeccoSubsystemCoupledTCN
except ImportError:
    GeccoSubsystemCoupledTCN = None

try:
    from .msl_coupled_tcn import MSLCoupledTCNStudent
except ImportError:
    MSLCoupledTCNStudent = None

try:
    from .universal_tcn import UniversalCyberPhysicalTCN, UniversalFoundationTCN_Pro
except ImportError:
    UniversalCyberPhysicalTCN = None
    UniversalFoundationTCN_Pro = None

STUDENT_MODELS = {
    "cnn_1d": CNN1DStudent,
    "tcn": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "tcn_residual": lambda in_channels, **kwargs: CNN1DStudent(in_channels=in_channels, use_residual=True, **kwargs),
    "stage_coupled_tcn": lambda **kwargs: StageCoupledTCNStudent(**kwargs),
    "smd_coupled_tcn": lambda **kwargs: SMDSubsystemCoupledTCN(**kwargs),
    "gecco_coupled_tcn": lambda **kwargs: GeccoSubsystemCoupledTCN(**kwargs),
    "msl_coupled_tcn": lambda **kwargs: MSLCoupledTCNStudent(**kwargs),
    "universal_tcn": lambda **kwargs: UniversalCyberPhysicalTCN(**kwargs),
    "universal_foundation_pro": lambda **kwargs: UniversalFoundationTCN_Pro(**kwargs),
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
    if name == "smd_coupled_tcn":
        return SMDSubsystemCoupledTCN(**kwargs)
    if name == "gecco_coupled_tcn":
        return GeccoSubsystemCoupledTCN(**kwargs)
    if name == "msl_coupled_tcn":
        return MSLCoupledTCNStudent(**kwargs)
    if name == "universal_tcn":
        return UniversalCyberPhysicalTCN(**kwargs)
    return STUDENT_MODELS[name](in_channels=in_channels, **kwargs)

__all__ = ["CNN1DStudent", "BiLSTMStudent", "TimesNetStudent", "StageCoupledTCNStudent", "SMDSubsystemCoupledTCN", "UniversalCyberPhysicalTCN", "build_student", "STUDENT_MODELS"]
