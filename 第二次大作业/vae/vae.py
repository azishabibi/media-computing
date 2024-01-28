import jittor as jt
from jittor import nn
# from jittor import init
import jittor.transform as transform
from jittor.dataset.mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import logging
import time
import datetime
from vae_model import VAE,vae_loss

jt.flags.use_cuda=1
seed = 42
jt.set_seed(seed)
np.random.seed(seed)
z1 = jt.randn(64, 64)

def generate_and_save_images(vae, num_images=64, save_path='result/big_image.png'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grid_size = int(num_images**0.5)  
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    with jt.no_grad():
        generated_images = vae.decoder(z1).view(num_images, 28, 28)

        for i in range(num_images):
            img = generated_images[i].numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')  

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path)
    plt.close(fig)  


transform = transform.Compose([
    transform.Gray(),  
    transform.ImageNormalize(mean=[0], std=[1]), 
    transform.ToTensor()
])
mnist = MNIST(train=True,transform=transform)
train_loader = mnist.set_attrs(batch_size=64, shuffle=True)
mnist_val = MNIST(train=False,transform=transform)
val_loader = mnist_val.set_attrs(batch_size=64, shuffle=False)

vae = VAE(input_dim=784, hidden_dims=[1024,512, 256, 128], z_dim=64)
optimizer = jt.optim.Adam(vae.parameters(), lr=0.001)
scheduler = jt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

beta=2 #change it if you want
num_epochs=200
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")

if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = f'logs/{current_time}_{beta}_training.log'

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

start_time = time.time()
min_val=1e18
for epoch in tqdm(range(num_epochs)):
    epoch_start = time.time()
    train_loss = 0
    for data, _ in train_loader:
        data = data.view(data.shape[0], -1)
        recon_batch, mu, log_var = vae(data)
        loss = vae_loss(recon_batch, data, mu, log_var,beta)
        optimizer.step(loss)
        train_loss += loss.item()

    train_loss /= len(train_loader)
    img_folder = f'vae_img_42/{beta}/'
    os.makedirs(img_folder, exist_ok=True)
    recon_batch_reshaped = recon_batch.view(-1, 28, 28).numpy()
    generate_and_save_images(vae,64,os.path.join(img_folder, f'batch_image_{epoch}.png'))
    val_loss = 0
    with jt.no_grad():
        for data, _ in val_loader:
            data = data.view(data.shape[0], -1)
            recon, mu, log_var = vae(data)
            val_loss += vae_loss(recon, data, mu, log_var).item()
    val_loss /= len(val_loader)
    if val_loss<min_val:
        min_val=val_loss
        jt.save(vae.state_dict(), f'./models/vae_model_{beta}.pkl')
    scheduler.step(val_loss)
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Epoch Time: {epoch_time}')
