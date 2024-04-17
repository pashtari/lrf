import os

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if os.path.isdir(root_dir):
            self.image_paths = os.listdir(root_dir)
        elif os.path.isfile(root_dir):
            self.image_paths = [root_dir]
            self.root_dir = ""

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # Since dataset doesn't have class labels, None is returned for the label
        return image.squeeze()
