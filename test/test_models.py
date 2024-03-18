import torch
from torchvision.models.resnet import resnet50, ResNet50_Weights
from skimage import data, img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt

from src import models

x = data.cat()
x = img_as_float(x)
x = resize(x, (224, 224), preserve_range=True)

plt.imshow(x)
plt.axis("off")
plt.show()

x = torch.tensor(x, dtype=torch.float32).permute(-1, 0, 1).unsqueeze(0)

interpolate_model = models.InterpolateModel(
    net=models.resnet50,
    num_classes=10,
    rescale=False,
    original_size=224,
    new_size=128,
    no_grad=True,
)
y = interpolate_model(x)
z = interpolate_model.transform(x)

z = torch.clip(z, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()
plt.imshow(z)
plt.axis("off")
plt.show()


patch_svd_model = models.PatchSVDModel(
    net=resnet50,
    num_classes=1000,
    patch_size=8,
    rank=10,
    domain="decompressed",
    weights=ResNet50_Weights.IMAGENET1K_V2,
)
y = patch_svd_model(x)
z = patch_svd_model.transform(x)

z = torch.clip(z, 0, 1).squeeze(0).permute(1, 2, 0).to(torch.float64).numpy()
plt.imshow(z)
plt.axis("off")
plt.show()