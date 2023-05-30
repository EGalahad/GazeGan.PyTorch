import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
])


class Dataset():
    def __init__(self, split="train", transform=train_transform, device="cuda") -> None:
        data_dir = "./data/dataset/CelebAGaze"
        data_dir = os.path.join(cur_dir, data_dir)
        # attr_0_txt = '0.txt'
        attr_1_txt = '1.txt'

        h, w = 256, 256

        self.split = split

        self.images_list, self.eye_pos = self.readfilenames(data_dir, attr_1_txt)

        self.transform = transform

        self.device = device
    
    def readfilenames(self, data_dir, filename):
        images_list = []
        eye_pos = []
        with open(os.path.join(data_dir, filename)) as file:
            for line in file.readlines():
                line = line.strip('\n')
                fields = line.split(' ', 5)
                if os.path.exists(os.path.join(data_dir, "1/"+fields[0]+".jpg")):
                    images_list.append(os.path.join(data_dir, "1/"+fields[0]+".jpg"))
                    eye_pos.append([int(value) for value in fields[1:5]])

        if self.split == "train":
            images_list = images_list[0:-100]
            eye_pos = eye_pos[0:-100]
        else:
            images_list = images_list[-100:]
            eye_pos = eye_pos[-100:]

        return images_list, eye_pos
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img = Image.open(self.images_list[idx])
        img = self.transform(img)
        return img, self.eye_pos[idx]
    
    def collate_fn(self, samples):
        # use torchvision to open all the images and turn into tensors
        crop_h, crop_w = 30, 50
        images, eye_pos = zip(*samples)
        # eye_pos: [batch_size, x_center, y_center]
        x_mask = torch.zeros((len(eye_pos), 1, 256, 256))
        for i in range(len(eye_pos)):
            x_mask[i, :, eye_pos[i][1]-crop_h//2:eye_pos[i][1]+crop_h//2, eye_pos[i][0]-crop_w//2:eye_pos[i][0]+crop_w//2] = 1
            x_mask[i, :, eye_pos[i][3]-crop_h//2:eye_pos[i][3]+crop_h//2, eye_pos[i][2]-crop_w//2:eye_pos[i][2]+crop_w//2] = 1
        return {
            "x": torch.stack(images),
            "x_mask": x_mask,
            "x_left_pos": torch.tensor([(pos[0], pos[1]) for pos in eye_pos]),
            "x_right_pos": torch.tensor([(pos[2], pos[3]) for pos in eye_pos])
        }