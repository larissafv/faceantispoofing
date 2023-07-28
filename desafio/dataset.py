import torch
from torchvision import transforms
import cv2

class IMG_Dataset:
    def __init__(self, X, y, batch_size, rng, device):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.rng = rng
        self.device = device
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                         std=[0.229, 0.224, 0.225])])
        
    def __getitem__(self, idx):
        """
        Returns:
            batch: List
            x: Tensor, shape ``[batch_size, 3, 224, 224]``
            y: Tensor, shape ``[batch_size]``
        """
        batch = self.rng.choice(idx, size = self.batch_size, replace = False)
        x = torch.stack([self.get_img(self.X[b]) for b in batch])
        y = torch.tensor([self.y[b] for b in batch])
        return batch, x.to(device = self.device, dtype = torch.float), y.to(self.device, dtype = torch.float)
    
    def get_img(self, path):
        """
        Arguments:
            path: String

        Returns:
            img: Tensor, shape ``[3, 224, 224]``
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img