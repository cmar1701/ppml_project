import torch
import tenseal
from PIL import Image
from glob import glob
from pandas import DataFrame
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize
from torchvision.transforms import InterpolationMode

class SkinCancerMNISTDataset:
    """A class to create a Dataset of skin lesion images and corresponding labels"""
    
    dx_to_integer_map = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df': 6,
        }

    def __init__(self, metadata: DataFrame, device: torch.device, tenseal_context: tenseal.Context | None = None):
        self.device = device
        self.tenseal_context = tenseal_context
        self.image_ids = metadata["image_id"]
        self.labels = metadata["dx"].apply(lambda x: self.dx_to_integer_map[x])

        self.image_id_to_filename_map = {}
        for image_filename in glob("data/HAM10000_images_part*/*.jpg"):
            # Image_id is the part after \\ and before.jpg (e.g. 'HAM10000_images_part_1\\ISIC_0024306.jpg')
            image_id = image_filename.rstrip(".jpg").split("\\")[-1]
            self.image_id_to_filename_map[image_id] = image_filename

    def __getitem__(self, idx):
        one_hot_encoding = torch.zeros(len(self.dx_to_integer_map), device=self.device)
        one_hot_encoding[self.labels.iloc[idx]] = 1.0
        label = one_hot_encoding
        
        image_id = self.image_ids.iloc[idx]
        image_filename = self.image_id_to_filename_map[image_id]

        if self.tenseal_context is None:
            image = read_image(image_filename).type(torch.FloatTensor).to(device=self.device)
        else:
            unencrypted_image = Image.open(image_filename)
            downsampled_unencrypted_image = Resize(40, interpolation=InterpolationMode.BILINEAR)(unencrypted_image)
            image = tenseal.ckks_tensor(self.tenseal_context, downsampled_unencrypted_image)

        return image, label

    def __len__(self):
        return len(self.image_ids)