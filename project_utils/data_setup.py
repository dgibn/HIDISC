import torch
from torch.utils.data import *
from torchvision import transforms
from torch.optim import Optimizer
from PIL import Image
import os
import pandas as pd
from project_utils.my_utils import *
from data.augmentations import get_transform
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
        # pos=self.data.iloc[idx,4]
        # positive_img=Image.open(pos).convert('RGB')
        # neg=self.data.iloc[idx,5]
        # negative_img=Image.open(neg).convert('RGB')
        # mask=self.data.iloc[idx, 6]
        
        if self.transform:
            image = self.transform(image)
            # positive_img=self.transform(positive_img)
            # negative_img=self.transform(negative_img)
        
        return image, numeric_label #, positive_img, negative_img ,mask

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

def mix_validation_dataloader(csv_file, batch_size, shuffle=True, num_workers=4, transform=None):
    # Create an instance of the OfficeHomeDataset
    
    dataset = DGCDDataset(csv_file=csv_file, transform=transform)
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def create_ViT_test_dataloaders(target_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list,transform, split: int) -> tuple[DataLoader, DataLoader]:
    
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

def create_ViT_train_dataloaders(source_domain: str, csv_dir_path: str, batch_size: int, selected_classes: list, transform, split: int) -> tuple[DataLoader, DataLoader]:
    domain_name=os.path.basename(source_domain)
    csv_folder_path = os.path.join(csv_dir_path, domain_name)
    os.makedirs(csv_folder_path, exist_ok=True)
    csv_path = create_target_csv(target_domain = source_domain,
                                 csv_dir_path = csv_folder_path,
                                 selected_classes = selected_classes,
                                 split = split)
    
    df = pd.read_csv(csv_path)
    filtered_df = df[df['continuous_numeric_label'].between(0, len(selected_classes)-1)]
    # Save the filtered DataFrame to a new CSV file
    csv_train_filename = f"{domain_name}_train.csv"
    train_csv_path = os.path.join(csv_folder_path, csv_train_filename)
    filtered_df.to_csv(train_csv_path, index=False) 

    train_dataset = DGCDDataset(csv_file = train_csv_path, transform = transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)

    return train_dataloader