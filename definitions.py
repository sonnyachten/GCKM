import os

import torch
from wcmatch import pathlib

torch.cuda.empty_cache()
Tensor = torch.Tensor
device = torch.device(
    "cuda" if torch.cuda.is_available() and not bool(os.environ.get("MPRKM_CPUONLY", False)) else "cpu")
# device = torch.device("cpu")
TensorType = torch.FloatTensor  # HalfTensor, FloatTensor, DoubleTensor
torch.set_default_tensor_type(TensorType)

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
OUT_DIR = pathlib.Path("~/out/gckm/").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = pathlib.Path("~/data").expanduser()
