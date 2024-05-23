

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

encoded_image_jpeg = lrf.pil_encode(image, format="JPEG", quality=20)


model = Wrapper(lrf.imf_decode, encoded_image_imf)
flops, macs, params = get_model_profile(model, input_shape=(10,), as_string=False)
print(f"imf decoder: {flops}")

model = Wrapper(lrf.pil_decode, encoded_image_jpeg)
flops, macs, params = get_model_profile(model, input_shape=(10,), as_string=False)
print(f"jpeg decoder: {flops}")
