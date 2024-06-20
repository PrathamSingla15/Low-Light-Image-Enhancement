import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from sklearn.model_selection import train_test_split
import cv2
import joblib
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
from tqdm import tqdm

with open('main_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    mdl = pickle.load(f)

def load_images(dir, name):
    return torch.stack([
        torch.from_numpy(np.array(Image.open(os.path.join(dir, f))))
        for f in tqdm(sorted(os.listdir(dir)), desc=f'Loading {name} images') if f.endswith('.png')
    ]).permute(0, 3, 1, 2) / 255

test_low_dir = './test/low/'

lo_t = load_images(test_low_dir, 'low_eval')

def Preprocessing_pipe(inp_lo):
    def dfq(q, b=0.75):
        return gaussian_kde(q, b * np.std(q))(np.linspace(0, 1, 255))

    def adjh(img, th):
        ih = np.histogram(img, bins=256, range=(0, 1))[0]
        ih, th = ih / np.sum(ih), th / np.sum(th)
        tf = np.interp(np.cumsum(ih), np.cumsum(th), np.linspace(0, 1, 255))
        return np.interp(img, np.linspace(0, 1, 256), tf)

    


    lx = inp_lo.reshape(-1, 400, 600)
    ix = np.array([np.histogram(img, bins=256, range=(0, 1))[0] for img in lx])
    out = mdl.predict(ix)
    print('Output produced')
    
    ni = []
    for i in range(len(inp_lo)):
        ims = []
        for k in range(3):
            hist = dfq(out[3*i + k])
            ims.append(adjh(inp_lo[i][k], hist))
        ni.append(ims)
    
    ni = torch.from_numpy(np.array(ni)).float()
    new_inp = [img.reshape(3, 400, 600) for img in ni]
    print("New input ready!")
    return(new_inp)


def hybrid_loss(outputs, targets, alpha=0.5, beta=0.5):
    l1 = F.mse_loss(outputs, targets)
    l2 = F.l1_loss(outputs, targets)
    return l1 + l2

class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_img = self.inputs[idx]
        target_img = self.targets[idx]
        return input_img, target_img
    

new_lo_t = Preprocessing_pipe(lo_t)

test_new_inp = new_lo_t
train_high = hi_t
test_dataset = MyDataset(train_new_inp, train_high)
test_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)




model.eval()
with torch.no_grad():
    test_loss = 0.0
    psnr_sum = 0.0
    num_batches = 0
    for low_img, normal_img in test_loader:
        low_img, normal_img = low_img.to(device), normal_img.to(device)
        outputs = model(low_img)
        loss = hybrid_loss(outputs,normal_img)
        test_loss += loss.item()

        # Calculate PSNR
        psnr_value = PSNR(outputs, normal_img)
        psnr_sum += psnr_value
        num_batches += 1

    average_psnr = psnr_sum / num_batches

    print(f"Test Loss: {test_loss / len(test_loader)}")
    print(f"Average PSNR: {average_psnr:.2f}")


num_images_to_show = len(test_loader)
for i, (images_low, images_normal) in enumerate(test_loader):
    if i >= num_images_to_show:
        break
    low_light_image = images_low[0]
    normal_light_image = images_normal[0]
    low_light_image = low_light_image.to(device)
    with torch.no_grad():
        output_image = model(low_light_image.unsqueeze(0))  
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(output_image.squeeze().cpu().numpy(), (1, 2, 0)))
    plt.title('Predicted Image')
    plt.axis('off')
    plt.show()
