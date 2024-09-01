# -*- coding: utf-8 -*-
"""
Created on Thu May  9 00:10:46 2024

@author: Nagham
"""

# 1. Apply Fourier transform on a gray image and show the
# transformed image before and after shifting.
import cv2
import numpy as np
from matplotlib import pyplot as plt


fig = plt.figure(figsize=(1,2))
img = cv2.imread(r"0002.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

forier=np.fft.fft2(gray)
forier_shift=np.fft.fftshift(forier)
fig.add_subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')
fig.add_subplot(1,2,2)
plt.imshow(np.log(np.abs(forier_shift)), cmap='gray')
plt.title('Forier')
plt.axis('off')




# 2. Design the following frequency domain filters, apply it on
# gray image and show image after applying the filters using
# inverse Fourier transform:

# • Ideal High pass filter
fig3 = plt.figure(figsize=(2,1))

rows, cols = gray.shape
center_row, center_col = rows // 2, cols // 2
cutoff_freq = 2
mask = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if distance > cutoff_freq:
                mask[i, j] = 1
ideal_LPF = forier_shift * mask
filtered_image = np.fft.ifftshift(ideal_LPF)
filtered_image = np.abs(np.fft.ifft2(filtered_image))
filtered_image = np.uint8(filtered_image)
fig3.add_subplot(2,1,1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')
fig3.add_subplot(2,1,2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
# • Gaussian High pass filter

# Assuming `gray` is the grayscale image
# Define the cutoff frequency D0
D0 = 10

# Get the shape of the grayscale image
M, N = gray.shape

# Create a filter H(u,v)
H = np.zeros((M, N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
        H[u, v] = np.exp(-D ** 2 / (2 * D0 ** 2))

# Create the high-pass filter HPF
HPF = 1 - H

# Apply the HPF to the Fourier transformed image
filtered_transform = forier_shift * HPF

# Shift back the filtered transform
filtered_transform_shifted = np.fft.ifftshift(filtered_transform)

# Inverse Fourier Transform to get the filtered image
filtered_image = np.abs(np.fft.ifft2(filtered_transform_shifted))

# Convert the filtered image to uint8 for visualization
filtered_image = np.uint8(filtered_image)

# Display the original and filtered images
fig5 = plt.figure(figsize=(2, 1))
fig5.add_subplot(2, 1, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

fig5.add_subplot(2, 1, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()


# • Ideal Low pass filter
fig2 = plt.figure(figsize=(2,1))

rows, cols = gray.shape
center_row, center_col = rows // 2, cols // 2
cutoff_freq = 2
mask = np.zeros((rows, cols), dtype=np.float32)
for i in range(rows):
    for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if distance <= cutoff_freq:
                mask[i, j] = 1
ideal_LPF = forier_shift * mask
filtered_image = np.fft.ifftshift(ideal_LPF)
filtered_image = np.abs(np.fft.ifft2(filtered_image))
filtered_image = np.uint8(filtered_image)
fig2.add_subplot(2,1,1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')
fig2.add_subplot(2,1,2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')


# • Gaussian Low pass filter
fig4 = plt.figure(figsize=(2,1))
                 
D0 = 10
M,N = gray.shape
H = np.zeros((M,N), dtype=np.float32)
for u in range(M):
    for v in range(N):
        D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
        H[u,v] = np.exp(-D**2/(2*D0*D0))

filtered_transform = forier_shift * H

filtered_image = np.fft.ifftshift(filtered_transform)
filtered_image = np.abs(np.fft.ifft2(filtered_image))
filtered_image= np.uint8(filtered_image)
fig4.add_subplot(2,1,1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')
fig4.add_subplot(2,1,2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
