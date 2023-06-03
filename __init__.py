import torch

import os
cur_dir = os.path.dirname(os.path.abspath(__file__))

def load_model(model_path, device):
    from model import GazeGan
    model = GazeGan()
    model.load_state_dict(torch.load(os.path.join(cur_dir, model_path), map_location=device))
    model.to(device)
    model.eval()
    return model

