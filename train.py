import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from model import GazeGan

from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--lr_g", type=float, default=0.00004)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--version", type=str, default="V2")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    train_batch_size = args.batch_size
    test_batch_size = 128
    num_epochs = args.num_epochs

    lr_d, lr_g = args.lr_d, args.lr_g

    if args.wandb:
        wandb.init(project="gaze-gan", config=args)

    torch.autograd.set_detect_anomaly(True)
    train_set = Dataset(split="train")
    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True, 
        num_workers=4, collate_fn=train_set.collate_fn)
    test_set = Dataset(split="test")
    test_loader = DataLoader(
        test_set, batch_size=test_batch_size, shuffle=True,
        num_workers=4, collate_fn=test_set.collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GazeGan().to(device)

    optimizer_d = torch.optim.Adam(model.Dx.parameters(), lr=lr_d)
    optimizer_g = torch.optim.Adam(model.Gx.parameters(), lr=lr_g)

    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=1, gamma=0.9)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=1, gamma=0.9)

    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []
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

                if args.wandb:
                    wandb.log({"train/G_loss": G_loss.item(), "train/D_loss": D_loss.item()})

                tepoch.set_description(f"Epoch {epoch}: G_loss: {sum(g_losses)/len(g_losses):.4f}, D_loss: {sum(d_losses)/len(d_losses):.4f}")
            
        scheduler_d.step()
        scheduler_g.step()
            
        test_g_losses = []  
        test_d_losses = []
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

                        if args.wandb:
                            wandb.log({"test/G_loss": G_loss.item(), "test/D_loss": D_loss.item()})

                        tepoch.set_description(f"Epoch {epoch}: G_loss: {sum(test_g_losses)/len(test_g_losses):.4f}, D_loss: {sum(test_d_losses)/len(test_d_losses):.4f}")
        
            torch.save(model.state_dict(), f"./checkpoints/model_{epoch}.pt")

if __name__ == "__main__":
    main()
