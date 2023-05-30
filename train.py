import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from model import GazeGan

from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm


def main():
    wandb.init(project="gaze-gan")

    torch.autograd.set_detect_anomaly(True)
    train_set = Dataset(split="train")
    train_loader = DataLoader(
        train_set, batch_size=32, shuffle=True, 
        num_workers=4, collate_fn=train_set.collate_fn)
    test_set = Dataset(split="test")
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=True,
        num_workers=4, collate_fn=test_set.collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GazeGan().to(device)

    model.load_state_dict(torch.load("./checkpoint.pt"))

    lr_d, lr_g = 0.0001, 0.00004
    optimizer_d = torch.optim.Adam(model.Dx.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(model.Gx.parameters(), lr=lr_g)

    EPOCH = 10
    for epoch in range(EPOCH):
        lam = 1 - 0.95 * (epoch / EPOCH)
        optimizer_d.param_groups[0]["lr"] = lr_d * lam
        optimizer_g.param_groups[0]["lr"] = lr_g * lam
        g_losses = []
        d_losses = []
        test_g_losses = []  
        test_d_losses = []
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, batch in enumerate(tepoch):
                model.train()
                x, x_mask, x_left_pos, x_right_pos = batch.values()

                x, x_mask, x_left_pos, x_right_pos = \
                    x.to(device), x_mask.to(device), x_left_pos.to(device), x_right_pos.to(device)
                
                G_loss, D_loss = model.get_loss(x, x_mask, x_left_pos, x_right_pos)

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()

                D_loss.backward(retain_graph=True)
                G_loss.backward()
                
                optimizer_d.step()
                optimizer_g.step()

                g_losses.append(G_loss.item())
                d_losses.append(D_loss.item())

                wandb.log({"train/G_loss": G_loss.item(), "train/D_loss": D_loss.item()})

                tepoch.set_description(f"Epoch {epoch}: G_loss: {sum(g_losses)/len(g_losses):.4f}, D_loss: {sum(d_losses)/len(d_losses):.4f}")
            
        if epoch % 1 == 0:
            # test
            model.eval()
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                    for i, batch in enumerate(tepoch):
                        x, x_mask, x_left_pos, x_right_pos = batch.values()

                        x, x_mask, x_left_pos, x_right_pos = \
                            x.to(device), x_mask.to(device), x_left_pos.to(device), x_right_pos.to(device)
                        
                        G_loss, D_loss = model(x, x_mask, x_left_pos, x_right_pos)

                        test_g_losses.append(G_loss.item())
                        test_d_losses.append(D_loss.item())

                        wandb.log({"test/G_loss": G_loss.item(), "test/D_loss": D_loss.item()})

                        tepoch.set_description(f"Epoch {epoch}: G_loss: {sum(test_g_losses)/len(test_g_losses):.4f}, D_loss: {sum(test_d_losses)/len(test_d_losses):.4f}")
        
            torch.save(model.state_dict(), f"./checkpoints/model_{epoch}.pt")

if __name__ == "__main__":
    main()
