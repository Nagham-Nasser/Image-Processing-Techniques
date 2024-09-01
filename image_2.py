# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:15:06 2024

@author: Nagham
"""

# 1.By using pillow library read an image and get the 
# following information:
# a)Type of image 
# b)Name of image 
# c) Mode of image 
# d)Size of image 
# e)Format of image
# f) Pixel’s values of image
import PIL
img = PIL.Image.open(r"0002.jpg")
#img.show()
print("Image type : ",type(img))
print("IMage Name : ",img.filename)
print("IMage Mode : ",img.mode)
print("IMage Name : ",img.size)
print("IMage Format : ",img.format)
#print("IMage Pixel : ",img)


# 2.By Using functions of pillow library perform the 
# following manipulations on the image, and plot all 
# images in the same figure:
# a)Crop the image.
# b)Resample the image using resize function and 
# reduce function.
import PIL
from matplotlib import pyplot as plt
import cv2

image = PIL.Image.open(r"0002.jpg")
cropped_image = image.crop((0,0,160,190))
fig1 = plt.figure(figsize=(7,7))
fig1.add_subplot(2,2,1)
plt.imshow(image)
fig1.add_subplot(2,2,2)
plt.imshow(cropped_image)

resized_image = image.resize((512,1024))
fig1.add_subplot(2,2,3)
plt.imshow(resized_image)


#3. Save the cropped image.
resized_image.save("cropped.jpg")


# Perform the following transformations using transpose 
# function and plot all images in the same figure:
# a)Flip the image left to right.
# b)Flip the image top to bottom.
# c)rotate the image 90,180 and 270.
# d)Transposes the rows and columns using the top-left 
# pixel as the origin.
# e)Transposes the rows and columns using the bottomleft pixel as the origin.
#5. Rotate image using rotate function.

import PIL

imag = PIL.Image.open(r"0002.jpg")
flipped1 = imag.transpose(PIL.Image.FLIP_LEFT_RIGHT)
flipped2 = imag.transpose(PIL.Image.FLIP_TOP_BOTTOM)
fig2 = plt.figure(figsize=(7,7))
fig2.add_subplot(2,2,1)
plt.imshow(flipped1)
fig2.add_subplot(2,2,2)
plt.imshow(flipped2)

angle = 180
rotate1 = imag.rotate(angle)
fig2.add_subplot(2,2,3)
plt.imshow(rotate1)

trans = imag.transpose(PIL.Image.TRANSPOSE)
fig2.add_subplot(2,2,3)
plt.imshow(trans)

trans = imag.transpose(PIL.Image.TRANSVERSE)
fig2.add_subplot(2,2,4)
plt.imshow(trans)



#6. Print the bands of image.
print(imag.getbands())

#7. Convert image to another modes.
grayscale=imag.convert("L")
#grayscale.show()
cmyk=imag.convert("CMYK")
#cmyk.show()

#. separate an image into its bands and plot each band individually
import cv2
(r,g,b)=imag.split()
fig3 = plt.figure(figsize=(7,7))
fig3.add_subplot(2,2,1)
plt.imshow(r,cmap="Reds")
fig3.add_subplot(2,2,2)
plt.imshow(g,cmap="Greens")
fig3.add_subplot(2,2,3)
plt.imshow(b,cmap="Blues")






    
    
    
    
    
    
    
    
    