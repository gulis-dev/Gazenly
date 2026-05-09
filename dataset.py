import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_all_datasets(split):
    image_list = []
    map_list = []
    
    # 1. SALICON
    salicon_img_dir = os.path.join(BASE_DIR, "data/salicon", split, "images")
    salicon_map_dir = os.path.join(BASE_DIR, "data/salicon", split, "maps")
    if os.path.exists(salicon_img_dir):
        files = os.listdir(salicon_img_dir)
        for f in files:
            if f.endswith(".jpg"):
                img_path = os.path.join(salicon_img_dir, f)
                map_path = os.path.join(salicon_map_dir, f.replace(".jpg", ".png"))
                image_list.append(img_path)
                map_list.append(map_path)
                
    # 2. ECdata
    if split in ["train", "test"]:
        list_file = os.path.join(BASE_DIR, "data/ECdata", f"{split}_list.txt")
        if os.path.exists(list_file):
            with open(list_file, "r") as file:
                content = file.read().strip()
                if content.startswith("[") and content.endswith("]"):
                    ids = content[1:-1].split(",")
                    for id_str in ids:
                        img_id = id_str.strip()
                        if not img_id: continue
                        img_path = os.path.join(BASE_DIR, "data/ECdata/ALLSTIMULI", f"{img_id}.jpg")
                        map_path = os.path.join(BASE_DIR, "data/ECdata/ALLFIXATIONMAPS", f"{img_id}_fixMap.jpg")
                        if os.path.exists(img_path):
                            image_list.append(img_path)
                            map_list.append(map_path)

    # 3. UEyes_dataset
    if split in ["train", "test"]:
        csv_path = os.path.join(BASE_DIR, "data/UEyes_dataset/image_types.csv")
        if os.path.exists(csv_path):
            df_ueyes = pd.read_csv(csv_path, sep=";")
            split_label = "Train" if split == "train" else "Test"
            if "Train/Test" in df_ueyes.columns:
                df_filtered = df_ueyes[df_ueyes["Train/Test"] == split_label]
                for idx, row in df_filtered.iterrows():
                    img_name = str(row["Image Name"]).strip()
                    img_path = os.path.join(BASE_DIR, "data/UEyes_dataset/images", img_name)
                    map_path = os.path.join(BASE_DIR, "data/UEyes_dataset/saliency_maps/fixmaps_3s", img_name)
                    if os.path.exists(img_path):
                        image_list.append(img_path)
                        map_list.append(map_path)

    data = {"image": image_list, "map": map_list}
    return pd.DataFrame(data)

class UnifiedDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            idx = 0
            
        img_path = self.df.iloc[idx]["image"]
        map_path = self.df.iloc[idx]["map"]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if os.path.exists(map_path):
                saliency_map = Image.open(map_path).convert("L")
            else:
                saliency_map = Image.new("L", image.size, color=0)
            
            image = self.transform(image)
            saliency_map = self.transform(saliency_map)
        except Exception as e:
            return self.__getitem__(0)
            
        return image, saliency_map

if __name__ == "__main__":
    df_train = load_all_datasets("train")
    print(f"Total training images: {len(df_train)}")
    dataset = UnifiedDataset(df_train)
    if len(dataset) > 0:
        img, smap = dataset[0]
        print("Image tensor shape:", img.shape)
        print("Map tensor shape:", smap.shape)
