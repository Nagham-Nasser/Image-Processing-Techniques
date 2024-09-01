# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:35:57 2024

@author: Nagham
"""

# 1.By using pillow library read an image and apply the 
# following filters:
import PIL
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(4,4))

original_img = PIL.Image.open(r"0002.jpg")
gray = original_img.convert('L')    
#  BLUR
blured = gray.filter(PIL.ImageFilter.BLUR())
fig.add_subplot(4,4,1)
plt.imshow(blured)
#  DETAIL
detailed = gray.filter(PIL.ImageFilter.DETAIL())
fig.add_subplot(4,4,2)
plt.imshow(detailed)
#  CONTOUDR
countored = gray.filter(PIL.ImageFilter.CONTOUR())
fig.add_subplot(4,4,3)
plt.imshow(countored)
#  EDGE_ENHANCE
edge_enhace = gray.filter(PIL.ImageFilter.EDGE_ENHANCE())
fig.add_subplot(4,4,4)
plt.imshow(edge_enhace)
#  EDGE_ENHANCE_MORE
edge_enhace_more = gray.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE())
fig.add_subplot(4,4,5)
plt.imshow(edge_enhace_more)
#  EMBOSS
edge_enhace_more = gray.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE())
fig.add_subplot(4,4,5)
plt.imshow(edge_enhace_more)
#  FIND_EDGES
find_edge = gray.filter(PIL.ImageFilter.FIND_EDGES())
fig.add_subplot(4,4,6)
plt.imshow(find_edge)
#  SMOOTH
smooth = gray.filter(PIL.ImageFilter.SMOOTH())
fig.add_subplot(4,4,7)
plt.imshow(smooth)
#  SMOOTH_MORE
smooth_more = gray.filter(PIL.ImageFilter.SMOOTH_MORE())
fig.add_subplot(4,4,8)
plt.imshow(smooth_more)
#  SHARPEN
sharpen = gray.filter(PIL.ImageFilter.SHARPEN())
fig.add_subplot(4,4,9)
plt.imshow(sharpen)
#  MaxFilter
maxfilter = gray.filter(PIL.ImageFilter.MaxFilter(5))
fig.add_subplot(4,4,10)
plt.imshow(maxfilter)
#  MedianFilter
medianfilter = gray.filter(PIL.ImageFilter.MedianFilter(7))
fig.add_subplot(4,4,11)
plt.imshow(medianfilter)
#  MinFilter
minfilter = gray.filter(PIL.ImageFilter.MinFilter(9))
fig.add_subplot(4,4,12)
plt.imshow(minfilter)

#  ModeFilter
mode = gray.filter(PIL.ImageFilter.ModeFilter(9))
fig.add_subplot(4,4,13)
plt.imshow(edge_enhace_more)
#  GaussianBlur
std=9
GaussianBlur = gray.filter(PIL.ImageFilter.GaussianBlur(std))
fig.add_subplot(4,4,14)
plt.imshow(GaussianBlur)
#  BoxBlur
BoxBlur = gray.filter(PIL.ImageFilter.BoxBlur(5))
fig.add_subplot(4,4,15)
plt.imshow(BoxBlur)
#  UnsharpMask
UnsharpMask = gray.filter(PIL.ImageFilter.UnsharpMask(radius=9,percent=75,threshold=5))
fig.add_subplot(4,4,16)
plt.imshow(UnsharpMask)



# 2-By Using ImageFilter. Kernel and apply any kernel to 
# the image.
kernal = [1,1,1,1,1,1,1,1,1]

filtered = gray.filter(PIL.ImageFilter.Kernel((3,3), kernal))
plt.show(filtered)


# using the following kernels, numpy module and 
# convolve function from SciPy module ,write a code to 
# apply the following kernels to an image.
kernel =[[1,2,1],
         [0,0,0],
         [-1,-2,-1]]
import numpy as np
from scipy.signal import convolve2d

image_array = np.array(gray)
# Apply convolution
convolved_image = convolve2d(image_array, kernel)
# Clip values to stay within the 0-255 range
convolved_image = np.clip(convolved_image, 0, 255)
# Convert back to PIL Image
convolved_image = PIL.Image.fromarray(convolved_image.astype(np.uint8))
convolved_image.show()














