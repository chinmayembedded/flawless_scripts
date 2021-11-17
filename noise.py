from skimage.util import random_noise
import torch 


def add_random_noise(image, pixel_noise_probability=0.2):
        img = image.cpu().detach().numpy()
        s_and_p = torch.tensor(random_noise(img, mode='s&p', salt_vs_pepper= pixel_noise_probability, clip=True))
        return s_and_p