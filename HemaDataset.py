from PIL import Image
import os
import numpy as np
import torch

from monai.networks import one_hot

class HemaDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_transform=None, target_transform=None, seg_entire_cell=True, split="train", return_meta=False):
        self.root = root
        self.split = split
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.seg_entire_cell = seg_entire_cell
        self.return_meta = return_meta
        self.define_split()

    def define_split(self):
        split_options = ["train", "val", "test", "test-ex", "test-in"]
        if self.split == "train":
            self.images_root = self.root + os.sep + "imagesTr"
            self.labels_root = self.root + os.sep + "labelsTr"
        elif self.split == "val":
            self.images_root = self.root + os.sep + "imagesVl"
            self.labels_root = self.root + os.sep + "labelsVl"
        elif self.split == "test-ex":
            self.images_root = self.root + os.sep + "imagesTs-External"
            self.labels_root = None
        elif self.split == "test-in" or self.split == "test":
            self.images_root = self.root + os.sep + "imagesTs-Internal"
            self.labels_root = None
        else:
            raise Exception(f"wrong split {self.split}, possible options are {split_options}")
        
        self.images = sorted([self.images_root + os.sep + image_name for image_name in os.listdir(self.images_root)]) 
        if self.labels_root != None:
            self.labels = sorted([self.labels_root + os.sep + label_name for label_name in os.listdir(self.labels_root)])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, indx):
        image = self.images[indx]
        image = Image.open(image)
                
        if self.labels_root != None:
            label = self.labels[indx]
            label = Image.open(label)
        else:
            label = None
        
        # Define meta variables
        meta = {}
        width, height = image.size
        meta["path"] = self.images[indx]
        meta["size"] = width
            
        seed = np.random.randint(2147483647) # make a seed with numpy generator
        if self.image_transform != None:
            np.random.seed(seed), torch.manual_seed(seed) 
            image = self.image_transform(image)

        if self.target_transform != None:
            np.random.seed(seed), torch.manual_seed(seed) 
            label = self.target_transform(label)
        
        if self.labels_root != None:
            # one hot encode the label
            label = one_hot(label.unsqueeze(0), num_classes=3).squeeze(0)
            if self.seg_entire_cell: # if we want to segment the entire cell as a class, we assign labels 1+2 as a new label 1.
                cyto = (label[1:2, :, :] == 1)
                nucleus = (label[2:3, :, :] == 1)
                entire_cell = cyto | nucleus
                label[1:2, :, :] = torch.where(entire_cell, torch.tensor(1.0, device=label.device), label[1:2, :, :])
        if self.return_meta:
            return image, label, meta
        return image, label