import torch

def load_model(model_path, device):
    from model import GazeGan
    model = GazeGan()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

