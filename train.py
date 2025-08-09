# train.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import ColorizationDataset
from model import UNetColorization
from config import *
import os
from tqdm import tqdm

def train():
    dataset = ColorizationDataset(data_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNetColorization().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Wrap dataloader in tqdm
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for i, (L, ab) in enumerate(loop):
            L, ab = L.to(device), ab.to(device)
            output_ab = model(L)

            loss = criterion(output_ab, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())  # Show current batch loss

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"checkpoints/unet_epoch{epoch+1}.pth")

if __name__ == "__main__":
    train()
