import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_split(split):
    folder = "data/salicon/" + split + "/images"
    files = os.listdir(folder)
    image_list = []
    for f in files:
        if f.endswith(".jpg"):
            image_list.append(f)
    image_list.sort()
    
    map_list = []
    for image in image_list:
        map_name = image.replace(".jpg", ".png")
        map_list.append(map_name)
        
    data = {"image": image_list, "map": map_list}
    return pd.DataFrame(data)

class SaliconDataset(Dataset):
    def __init__(self, df, split):
        self.df = df
        self.split = split
        self.image_dir = "data/salicon/" + split + "/images/"
        self.map_dir = "data/salicon/" + split + "/maps/"
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        # i dont know why but sometimes it shows error here if empty?
        return len(self.df)

    def __getitem__(self, idx):
        # sometimes idx is weird so i do this
        if idx >= len(self.df):
            idx = 0
            
        img_name = self.df.iloc[idx]["image"]
        map_name = self.df.iloc[idx]["map"]
        
        img_path = self.image_dir + img_name
        map_path = self.map_dir + map_name
        
        # if one file is missing it crashes everything so i add try
        try:
            image = Image.open(img_path).convert("RGB")
            saliency_map = Image.open(map_path).convert("L")
            
            image = self.transform(image)
            saliency_map = self.transform(saliency_map)
        except Exception as e:
            print(f"problem with {img_name}, skipping?")
            # just return first one to not break the loop
            return self.__getitem__(0)
            
        return image, saliency_map

if __name__ == "__main__":
    df_train = load_split("train")
    dataset = SaliconDataset(df_train, "train")
    
    img, smap = dataset[0]
    print("Image tensor shape:", img.shape)
    print("Map tensor shape:", smap.shape)
