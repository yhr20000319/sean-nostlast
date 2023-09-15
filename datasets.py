import glob
import random
import os
import numpy as np

import jittor as jt
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image
import json

class ImageDataset(Dataset):
    def __init__(self, root, transform_label, transform_img, mode="train"):
        super().__init__()
        self.transform_label = transform.Compose(transform_label)
        self.transform_img = transform.Compose(transform_img)
        self.mode = mode
        #if self.mode == 'train':
        #    self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
        self.files = sorted(glob.glob(os.path.join(root, "imgs") + "/*.*"))
        
        self.labels = sorted(glob.glob(os.path.join(root,"labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        
        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.75:
                k = (np.random.random() + 3)*0.25
                x = np.random.random()*1024*(1-k)
                y = np.random.random()*768*(1-k)
                img_A = img_A.crop((x, y, x+1024*k, y+768*k)) 
                img_B = img_B.crop((x, y, x+1024*k, y+768*k))   
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transform_img(img_A)
        else:
            img_A = Image.open(self.files[index % len(self.files)])
            img_A = self.transform_img(img_A)
            #img_A = np.empty([1])
        img_B = self.transform_label(img_B) * 255
        return img_A, img_B, photo_id
    
    
class TestImageDataset(Dataset):
    def __init__(self, root,ref,transform_label, transform_img):
        super().__init__()
        self.transform_label = transform.Compose(transform_label)
        self.transform_img = transform.Compose(transform_img)
        self.labels = sorted(glob.glob(root + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        self.ref=ref
        print(f"from test split load {self.total_len} images.")
        with  open('./label_to_img.json', 'r') as  f:  
            self.ref_dict  =  json.load(f)

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        #img_A = np.empty([1])
        img_B = self.transform_label(img_B) * 255
        
        label_file  =  label_path.split("/")[-1]  
        img_file  =  self.ref_dict[label_file].replace(".png",".jpg")  
        #image_path  =  "/".join(["datasets/train/imgs", img_file]) 
        image_path  =  "/".join([self.ref, img_file])
        img_A = Image.open(image_path)
        img_A = self.transform_img(img_A)
        
        #image_path_labels  =  "/".join(["datasets/train/labels", img_file])
        img_A_label = Image.open(label_path)
        img_A_label  = self.transform_label(img_A_label)
        
        return img_A, img_B,img_B, photo_id
