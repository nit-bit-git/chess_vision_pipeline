from .transforms import resize, normalize
import torch 


def testfn():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)