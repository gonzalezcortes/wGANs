import matplotlib.pyplot as plt
import numpy as np
import torchvision

class Plots:
    @staticmethod
    def losses(generator_loss_list, critic_loss_list):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Critic Loss During Training")
        plt.plot(generator_loss_list,label="Generator")
        plt.plot(critic_loss_list,label="Critic")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('images/Generator_and_Critic_Loss.png')
        plt.show()

        

    @staticmethod
    def imshow(img, title):
        img = img.detach().cpu() / 2 + 0.5 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        

    @staticmethod
    def plot_real_vs_fake(real_images, fake_images):
        real_images = real_images.view(real_images.size(0), 1, 28, 28)
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

        real_images = real_images[:16]
        fake_images = fake_images[:16]

        plt.figure(figsize=(10,5))

        plt.subplot(1, 2, 1)
        Plots.imshow(torchvision.utils.make_grid(real_images), title='Real Images')

        plt.subplot(1, 2, 2)
        Plots.imshow(torchvision.utils.make_grid(fake_images), title='Fake Images')

        plt.savefig('images/plot_real_vs_fake.png')
        plt.show()
