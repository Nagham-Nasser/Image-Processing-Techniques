# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:56:02 2024

@author: Nagham
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import PIL

# 1.By using open cv library read an image and apply the
# following noises:
# Gaussian Noise
def add_gaussian_noise(image,mean,std):
    noise = np.random.normal(mean,std,image.shape).astype(np.uint8)
    noisy_image = cv2.add(noise,image)
    return noisy_image
    
image = cv2.imread("0002.jpg")
noisy_img = add_gaussian_noise(image, 0,25)
plt.imshow(noisy_img)
plt.title('Gaussian Noise')
plt.axis('off')          
                
# Salt and Paper Noise
def add_salt_and_paper_noise(image, noisy_ratio):
    noisy_image = image.copy()
    h,w,c = noisy_image.shape
    noisy_pixels = int( h * w* noisy_ratio)
    for _ in range (noisy_pixels):
        row,colm = np.random.randint(0,h), np.random.randint(0,w)
        if np.random.rand() < 0.5:
            noisy_image[row,colm] = [0,0,0]
        else:
            noisy_image[row,colm] = [255,255,255]
    return noisy_image

noisy_img = add_salt_and_paper_noise(image,0.5)
plt.imshow(noisy_img)
plt.title('Salt and Paper Noise')
plt.axis('off')
 
# Random Noise
def add_random_noise(image,intensity):
    noisy_image = image.copy()
    noise = np.random.randint(-1*intensity, intensity +1,image.shape)
    noisy_image = np.clip((image + noisy_image), 0, 255).astype(np.uint8)
    return noisy_image
noisy_img = add_random_noise(image, 100)
plt.imshow(noisy_img) 
plt.title('Random Noise')
plt.axis('off')
# 3- using pillow library read an image and compress it.
def image_compression(image,comp_ratio):
    h,w = image.size
    row,colm =int( h/comp_ratio),int(w/comp_ratio)
    new_size = (row,colm)
    resized_image = image.resize(new_size)
    resized_image.save("comp.jpg",optimize = False,quality = 50)
    comp_size = os.path.getsize("comp.jpg")
    print("comp size : ",comp_size)
    

image = PIL.Image.open("0002.jpg")
print("orig_size : ",os.path.getsize("0002.jpg"))
image_compression(image,2)           
            