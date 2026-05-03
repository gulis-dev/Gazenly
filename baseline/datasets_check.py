import pandas as pd
import os

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
        
    data = {
        "image": image_list,
        "map": map_list
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    result = load_split("train")
    print(result.head(10))
    print(result.count)
