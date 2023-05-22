import sys

sys.path.append('src/')

import torch
import torch.optim as optim
import numpy as np

from OpenData import DataLoaderCreator
from wGAN import Generator, Critic, Constraint


from PlottingFx import Plots

load_data = DataLoaderCreator()
trainloader = load_data.trainloader
testloader = load_data.testloader


seed = 7
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


latent_dim = 100 
n_critic = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lambda_gp = 10

generator = Generator(latent_dim).to(device)
critic = Critic().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.9))

epochs = 100

generator_loss_list, critic_loss_list = [], []

for epoch in range(epochs):
    for images, _ in trainloader:
        batch_size = images.shape[0]

        for _ in range(n_critic):
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_r = generator(noise).detach()

            real_imgs = images.view(batch_size, -1).to(device)
            critic_real = critic(real_imgs)
            critic_fake = critic(fake_r)

            gradient_penalty = Constraint.compute_gradient_penalty(critic, real_imgs.data, fake_r.data)
            critic_loss = -torch.mean(critic_real) + torch.mean(critic_fake) + lambda_gp * gradient_penalty

            optimizer_C.zero_grad()
           
            critic_loss.backward()
            optimizer_C.step()

        noise = torch.randn(batch_size, latent_dim).to(device)
        fake_r = generator(noise)
        generator_loss = -torch.mean(critic(fake_r))

        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

    generator_loss_list.append(generator_loss.item())
    critic_loss_list.append(critic_loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: [Critic loss: {critic_loss.item()}] [Generator loss: {generator_loss.item()}] ")

Plots.losses(generator_loss_list, critic_loss_list)

real_images, _ = next(iter(trainloader))
noise_z = torch.randn(batch_size, latent_dim).to(device)
fake_images = generator(noise_z)

Plots.plot_real_vs_fake(real_images, fake_images)