import cv2
import numpy as np
import glob

# Get all image filenames in a folder
img_filenames = glob.glob('path/to/images/*.jpg')

# Load the first image to get dimensions
img = cv2.imread(img_filenames[0])
height, width, layers = img.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width,height))

# Loop through all images and write to video
for filename in img_filenames:
    img = cv2.imread(filename)
    out.write(img)

# Release the VideoWriter object and close all windows
out.release()
cv2.destroyAllWindows()

###############################################################################
#BOUNDING BOX
import cv2

# Load the image
img = cv2.imread('path/to/image.jpg')

# Define the coordinates of the rectangle
x, y, w, h = 100, 100, 200, 150

# Draw the rectangle
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##################################################################################
#SELECT MANUALLY THE FACE
import cv2

# Load the image
img = cv2.imread('path/to/image.jpg')

# Display the image and select ROI
r = cv2.selectROI(img)

# Draw the rectangle
x, y, w, h = r
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with the bounding box
cv2.imshow('Image with Bounding Box', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##############################################################################