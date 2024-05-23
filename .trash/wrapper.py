

import torch

from deepspeed.profiling.flops_profiler import get_model_profile

from torch import nn
from torchvision.transforms import v2
from skimage.io import imread
from fvcore.nn import FlopCountAnalysis

import lrf


class Wrapper(nn.Module):
    def __init__(self, fun, *args) -> None:
        super().__init__()
        self.fun = fun
        self.args = args

    def forward(self, x):
        return self.fun(*self.args)


image = imread("./data/kodak/kodim19.png")
# Transform the input image
transforms = v2.Compose([v2.ToImage()])
image = torch.tensor(transforms(image))

encoded_image_imf = lrf.imf_encode(
    image,
    color_space="YCbCr",
    scale_factor=(0.5, 0.5),
    quality=(6, 3, 3),
    patch=True,
    patch_size=(8, 8),
    bounds=(-16, 15),
    dtype=torch.int8,
    num_iters=10,
    verbose=False,
)
<<<<<<< HEAD

encoded_image_jpeg = lrf.pil_encode(image, format="JPEG", quality=20)


model = Wrapper(lrf.imf_decode, encoded_image_imf)
flops = FlopCountAnalysis(model, ())
print(f"imf {flops.total()}")

flops, macs, params = get_model_profile(model, args=(), as_string=False)
=======
>>>>>>> db5d22bf48d24a3759e60a0b86f20de204c61661

encoded_image_jpeg = lrf.pil_encode(image, format="JPEG", quality=20)


model = Wrapper(lrf.imf_decode, encoded_image_imf)
flops, macs, params = get_model_profile(model, input_shape=(10,), as_string=False)
print(f"imf decoder: {flops}")

model = Wrapper(lrf.pil_decode, encoded_image_jpeg)
<<<<<<< HEAD
flops = FlopCountAnalysis(model, ())
print(f"jpeg {flops.total()}")
=======
flops, macs, params = get_model_profile(model, input_shape=(10,), as_string=False)
print(f"jpeg decoder: {flops}")
>>>>>>> db5d22bf48d24a3759e60a0b86f20de204c61661
