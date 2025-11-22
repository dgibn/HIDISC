import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from project_utils.my_utils import create_target_csv
from torch.utils.data import Dataset, DataLoader

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]
        
class DGCDDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]        
        image = Image.open(image_path).convert('RGB')
        numeric_label = self.data.iloc[idx, 3]
        
        if self.transform:
            image = self.transform(image)
        
        return image, numeric_label

class CombinedDomainDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path (str): Path to the combined CSV file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = int(row['numeric_label'])
        domain_label = int(row['domain_label'])

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, domain_label

def get_combined_dataloader(csv_path, batch_size=64, shuffle=True, num_workers=4, transform=None):
    """
    Args:
        csv_path (str): Path to the combined CSV file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of subprocesses for data loading.
        transform: Torchvision transform pipeline.

    Returns:
        DataLoader
    """
    dataset = CombinedDomainDataset(csv_path=csv_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def create_test_dataloaders(target_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list,transform, split: int) -> tuple[DataLoader, DataLoader]:
    target_domain = os.path.join("/users/student/pg/pg23/vaibhav.rathore/datasets/",target_domain)
    path = create_target_csv(target_domain = target_domain,
                             csv_dir_path = csv_dir_path,
                             selected_classes = selected_classes,
                             split=split)
    # Create Datasets and Dataloaders for both the additional CSV files
    target_Dataset = DGCDDataset(csv_file=path, transform=transform)

    target_Dataloader = DataLoader(dataset=target_Dataset,
                                         batch_size=batch_size,
                                         shuffle=True,num_workers=8)

    return target_Dataloader
