from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from utils import rgb_to_lab
from config import num_train_images  # NEW

class ColorizationDataset(Dataset):
    def __init__(self, image_folder, image_size):
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]
        
        # NEW: limit number of images if specified
        if num_train_images is not None:
            image_files = image_files[:num_train_images]
        
        self.image_paths = [os.path.join(image_folder, f) for f in image_files]
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.resize(img)  
        img_np = np.array(img) / 255.0

        L, ab = rgb_to_lab(img_np)
        L_tensor = self.to_tensor(L)
        ab_tensor = self.to_tensor(ab)

        return L_tensor, ab_tensor
