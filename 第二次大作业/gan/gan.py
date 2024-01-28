import jittor as jt
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging
import time
import datetime
jt.flags.use_cuda = 1 if jt.has_cuda else 0 

epoch = 500
batch_size = 512
learning_rate = 0.0001
z_size = 64
hidden_size = 256
transform = transform.Compose([
    transform.Gray(),  
    transform.ImageNormalize(mean=[0], std=[1]),  
    transform.ToTensor()
])

mnist_train = MNIST(train=True, transform=transform, download=True)
train_loader = mnist_train.set_attrs(batch_size=batch_size, shuffle=True, drop_last=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, 784),
            nn.Tanh()
        )

    def execute(self, z):
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        return out.view(batch_size, 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def execute(self, x):
        x = x.view(batch_size, -1)
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)

generator = Generator()
discriminator = Discriminator()

loss_func = nn.MSELoss()
gen_optim = jt.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999))
dis_optim = jt.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

ones_ = jt.ones(batch_size, 1)
zeros_ = jt.zeros(batch_size, 1)
if not os.path.exists('logs'):
    os.makedirs('logs')
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
log_filename = f'logs/{current_time}_training.log'

logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

for i in tqdm(range(epoch)):
    start_time = time.time()
    gen_loss_sum=0
    dis_loss_sum=0
    for j, (image, _) in enumerate(train_loader):
        #discriminator training
        z = jt.array(np.random.normal(0,0.1, size=(batch_size, z_size)))
        fake_image = generator(z)
        dis_real = discriminator(image)
        dis_fake = discriminator(fake_image.detach())
        dis_loss_real = loss_func(dis_real, ones_)
        dis_loss_fake = loss_func(dis_fake, zeros_)
        dis_loss = jt.sum(dis_loss_real) + jt.sum(dis_loss_fake)
        dis_optim.step(dis_loss)
        dis_loss_sum+=dis_loss.item()

        #generator training
        z = jt.array(np.random.normal(0,0.1, size=(batch_size, z_size)))
        fake_image = generator(z)
        output = discriminator(fake_image)
        gen_loss = jt.sum(loss_func(output, ones_))
        gen_optim.step(gen_loss)
        gen_loss_sum+=gen_loss.item()
        jt.save(generator.state_dict(), './models/generator.pkl')
        jt.save(discriminator.state_dict(), './models/discriminator.pkl')
    dis_loss_sum=dis_loss_sum/len(train_loader)
    gen_loss_sum=gen_loss_sum/len(train_loader)
    logging.info(f"Epoch: {i}, D Loss: {dis_loss_sum}, G Loss: {gen_loss_sum}")
    with jt.no_grad():
        test_z = jt.array(np.random.normal(0,0.1, size=(batch_size, z_size)))
        generated = generator(test_z)
        generated_np = generated[:25].numpy()
        fig, axs = plt.subplots(5, 5, figsize=(5, 5))
        for j, ax in enumerate(axs.flatten()):
            ax.imshow(generated_np[j].transpose(1, 2, 0) * 0.5 + 0.5, cmap='gray')
            ax.axis('off')
        plt.savefig(f"./generated_images/gen_{i}.png")
        plt.close()
