import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import load_split, SaliconDataset
from model import SaliencyModel

def train():
    batch_size = 4
    epochs = 1
    learning_rate = 0.0001
    
    # currently i working on macbooik
    device = torch.device("cpu")
    print("i am using:", device)
    
    df_train = load_split("train")
    
    # 10000 is way too much for now
    df_small = df_train.head(100)
    
    train_dataset = SaliconDataset(df_small, "train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = SaliencyModel().to(device)
    
    # mse loss is good for comparing two images (pixels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for i, (images, maps) in enumerate(train_loader):
            images = images.to(device)
            maps = maps.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, maps)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} done! Avg loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("everything saved to model.pth")

if __name__ == "__main__":
    train()
