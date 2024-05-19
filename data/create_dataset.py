import torch
from glob import glob
from torchvision.io import read_image


class SkinCancerMNISTDataset:
    """A class to create a Dataset of skin lesion images and corresponding labels"""
    
    def __init__(self, df):
        self.dx_to_integer_map = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df': 6,
        }
        
        self.image_ids = df["image_id"]
        self.labels = df["dx"].apply(lambda x: self.dx_to_integer_map[x])

        self.image_id_to_filename_map = {}
        for image_filename in glob("data/HAM10000_images_part*/*.jpg"):
            # Image_id is the part after \\ and before.jpg (e.g. 'HAM10000_images_part_1\\ISIC_0024306.jpg')
            image_id = image_filename.rstrip(".jpg").split("\\")[-1]
            self.image_id_to_filename_map[image_id] = image_filename

    def __getitem__(self, idx):
        one_hot_encoding = torch.zeros(len(self.dx_to_integer_map))
        one_hot_encoding[self.labels.iloc[idx]] = 1.0
        label = one_hot_encoding
        
        image_id = self.image_ids.iloc[idx]
        image_filename = self.image_id_to_filename_map[image_id]
        image = read_image(image_filename).type(torch.FloatTensor)

        return image, label

    def __len__(self):
        return len(self.image_ids)