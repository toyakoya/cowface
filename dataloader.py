
# import random
# from pathlib import Path
# from typing import List, Tuple, Dict

# from torch.utils.data import Dataset,DataLoader
# from torchvision import transforms
# from PIL import Image

# TRAIN_DIR = Path("./train")
# IMAGE_EXTS= {".jpg"}
# RANDOM_SEED = 42
# TRAIN_RATIO = 0.8
# BATCH_SIZE = 32
# NUM_WORKERS = 8
# IMG_SIZE =512
# random.seed(RANDOM_SEED)



# def get_cow_folders(root:Path) -> List[Path]:
#     return [p for p in sorted(root.iterdir()) if p.is_dir()]
# def collect_image_paths(folder:Path) -> List[Path]:
#     return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
# def train_val_split(cow_folders:List[Path],train_ratio:float=0.8,seed:int=42) -> Tuple[List[Path],List[Path]]:
#     rng=random.Random(seed)
#     folders=cow_folders.copy()
#     rng.shuffle(folders)
#     split=int(len(folders)*train_ratio)
#     return folders[:split],folders[split:]
# class CowImageDataset(Dataset):
#     """
#     return: image_tensor, label_idx, label_name, image_path
#     label_idx: 从0开始的整数类索引
#     label_name: 文件夹名
#     """
#     def __init__(self,folders:List[Path],transform=None):
#         self.transform=transform
#         self.samples=[]
#         self.label_to_idx={}
#         for idx, folder in enumerate(sorted(folders)):
#             label_name=folder.name
#             self.label_to_idx[label_name]=idx
#             img_paths=collect_image_paths(folder)
#             for p in img_paths:
#                 self.samples.append((p,idx,label_name))
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         img_path,label_idx,label_name=self.samples[idx]
#         img=Image.open(img_path).convert('RGB')
#         if self.transform:
#             img=self.transform(img)
#         return img,label_idx,label_name,str(img_path)

# def make_dataloaders(root_dir:Path,img_size:int=512,batch_size:int=32,num_workers:int=4,train_ratio:float=0.8,seed:int=42):
#     cow_folders = get_cow_folders(root_dir)
#     train_folders ,val_folders= train_val_split(cow_folders,train_ratio,seed)
#     # train_folders=cow_folders
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(img_size,scale=(0.8,1.0)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.1,0.1,0.1,0.02),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize(int(img_size*1.14)),
#         transforms.CenterCrop(img_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#     ])
#     train_dataset = CowImageDataset(train_folders,train_transform)
#     val_dataset = CowImageDataset(val_folders,val_transform)
#     train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
#     val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
#     meta={
#         "num_classes":len(train_dataset.label_to_idx),
#         "train_num_classes":len(train_dataset.label_to_idx),
#         "val_num_classes":len(val_dataset.label_to_idx),
#         "train_label_to_idx":train_dataset.label_to_idx,
#         "val_label_to_idx":val_dataset.label_to_idx,
#         "train_dataset":train_dataset,
#         "val_dataset":val_dataset,
#     }
#     return train_loader,val_loader,meta
#     # return train_loader,meta

# def get_dataloader():
#     train_loader,val_loader,meta=make_dataloaders(
#         root_dir=TRAIN_DIR,
#         img_size=IMG_SIZE,
#         batch_size=BATCH_SIZE,
#         num_workers=NUM_WORKERS,
#         train_ratio=TRAIN_RATIO,
#         seed=RANDOM_SEED
#     )
#     return train_loader,val_loader,meta

import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# ------------------- 配置常量（按需修改） -------------------
TRAIN_DIR = Path("./train")   # 识别训练集根目录（子文件夹为每头牛）
VAL_DIR = Path("./val")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
NUM_WORKERS = 8
IMG_SIZE = 224
# ---------------------------------------------------------

random.seed(RANDOM_SEED)
label_to_idx = {}
def get_cow_folders(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]

def collect_image_paths(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

# def train_val_split(cow_folders: List[Path], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Path], List[Path]]:
#     rng = random.Random(seed)
#     folders = cow_folders.copy()
#     rng.shuffle(folders)
#     split = int(len(folders) * train_ratio)
#     return folders, folders[split:]

class CowImageDataset(Dataset):
    """
    返回：image_tensor, label_idx, label_name, image_path
    label_idx: 从0开始的整数类索引
    label_name: 文件夹名
    可选参数 extra_augment 预留用于返回多视图（当前未启用）
    """
    def __init__(self, folders: List[Path], transform=None, type="train",extra_augment: int = 0):
        self.transform = transform
        self.extra_augment = extra_augment
        self.samples = []  # (path, label_idx, label_name)
        # self.label_to_idx = {}
        # self.label_to_idx = label_to_idx if label_to_idx is not None else {}
        for idx, folder in enumerate(sorted(folders)):
            label_name = folder.name
            if type=="train":
                label_to_idx[label_name] = idx
            img_paths = collect_image_paths(folder)
            for p in img_paths:
                if type=="train":
                    self.samples.append((p, idx, label_name))
                else:
                    self.samples.append((p, label_to_idx[label_name], label_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, label_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)
        # 如果需要返回多视图，可扩展这里
        return img_t, label_idx, label_name, str(img_path)

def make_transforms(img_size: int = 512):
    # 更强的训练增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1,0.02)]),
        # transforms.RandomGrayscale(p=0.02),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def make_dataloaders(root_dir: Path,
                     img_size: int = 512,
                     batch_size: int = 32,
                     num_workers: int = 4,
                     train_ratio: float = 0.8,
                     seed: int = 42):
    """
    返回 train_loader, val_loader, meta

    """
    # cow_folders = get_cow_folders(root_dir)
    # train_folders, val_folders = train_val_split(cow_folders, train_ratio, seed)
    train_folders = get_cow_folders(TRAIN_DIR)
    val_folders =get_cow_folders(VAL_DIR)
    train_transform, val_transform = make_transforms(img_size)

    train_dataset = CowImageDataset(train_folders, transform=train_transform,type="train")
    val_dataset = CowImageDataset(val_folders, transform=val_transform,type="val")

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    meta = {
        "num_classes": len(train_dataset),
        "train_num_classes": len(label_to_idx),
        "val_num_classes": len(val_dataset),
    }
    return train_loader, val_loader, meta

def get_dataloader():
    return make_dataloaders(
        root_dir=TRAIN_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
        use_weighted_sampler=True
    )

# quick test snippet when run directly
if __name__ == "__main__":
    tr, va, meta = get_dataloader()
    print("Train batches:", len(tr), "Val batches:", len(va),"meta:",meta)
    batch = next(iter(tr))
    imgs, labels_idx, labels_name, paths = batch
    print("imgs.shape:", imgs.shape)
    print("labels example:", labels_idx[:8])

    batch = next(iter(va))
    imgs, labels_idx, labels_name, paths = batch
    print("imgs.shape:", imgs.shape)
    print("labels example:", labels_idx[:8])