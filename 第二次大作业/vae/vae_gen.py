import os
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn

from vae_model import VAE

model_path = 'vae_model.pkl'
vae = VAE(input_dim=784, hidden_dims=[1024, 512, 256, 128], z_dim=64)
vae.load_state_dict(jt.load(model_path)) 
vae.eval()

os.makedirs('result', exist_ok=True)

num_images = 64  # Adjust this to change the number of images


fig, axes = plt.subplots(8, 8, figsize=(10, 10))  
axes = axes.flatten()

with jt.no_grad():
    for i in range(num_images):
        z = jt.randn(1, 64) #change 64 to your z_dim
        generated_image = vae.decoder(z).view(28, 28) 
        img = generated_image.numpy()

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off') 

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('result/big_image.png')
plt.close(fig)
