import os
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader

class GTSRBDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # thư mục gốc chứa ảnh (VD: /home/.../versions/1/)
        self.transforms = transforms

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load ảnh
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        # Lấy bounding box và nhãn
        boxes = [[row["Roi.X1"], row["Roi.Y1"], row["Roi.X2"], row["Roi.Y2"]]]
        boxes = torch.tensor(boxes, dtype=torch.float32)

        labels = torch.tensor([row["ClassId"]], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.data)

def get_dataloader(batch_size, dataset_root="dataset/"):

    train_transform = T.Compose([
        T.Resize((256, 512)),
        T.RandomRotation(degrees=10),                      # Xoay ảnh ±10 độ
        T.RandomHorizontalFlip(),                          # Lật ngẫu nhiên theo chiều ngang
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Dịch chuyển ảnh ±5% theo cả chiều ngang và dọc
        T.ColorJitter(brightness=0.2, contrast=0.2),       # Điều chỉnh độ sáng và độ tương phản
        T.RandomPerspective(distortion_scale=0.1, p=0.5),  # Biến dạng phối cảnh nhẹ
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_transform = T.Compose([
        T.Resize((256, 512), interpolation=Image.NEAREST),
        T.ToTensor()
    ])

    train_dataset = GTSRBDataset(
        csv_file=os.path.join(dataset_root, "Train/Train.csv"),
        root_dir=dataset_root,
        transforms=train_transform
    )

    total_size = len(train_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    test_dataset = GTSRBDataset(
        csv_file=os.path.join(dataset_root, "Test/Test.csv"),
        root_dir=dataset_root,
        transforms=test_transform
    )

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    return train_loader, val_loader, test_loader