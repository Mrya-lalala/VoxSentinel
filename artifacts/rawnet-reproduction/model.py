"""Official RawNet3 instantiated with a scalar binary classification output."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "dependencies"), str(ROOT / "upstream/python/RawNet3")]
from models.RawNet3 import RawNet3
from models.RawNetBasicBlock import Bottle2neck


def create_model():
    return RawNet3(
        Bottle2neck, model_scale=8, context=True, summed=True,
        encoder_type="ECA", nOut=1, out_bn=False, sinc_stride=10,
        log_sinc=True, norm_sinc="mean", grad_mult=1,
    )
