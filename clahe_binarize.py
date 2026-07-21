# apply clahe to all original images (only needs to be run once)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_folder = 'C:\\Users\\megan\\flies\\activity_analysis\\avg'
inputs = [os.path.join(input_folder, f) for f in sorted(os.listdir(input_folder)) if f.endswith(".tif")]
output = 'clahe_binarize'
os.makedirs(output, exist_ok=True)

fig, axes = plt.subplots(3,len(inputs))
for i in range(len(inputs)):
    image = cv2.imread(inputs[i], cv2.IMREAD_UNCHANGED)
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0,i].imshow(image, cmap='gray')
    axes[0,i].axis('off')

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16,16))
    clahe_image = clahe.apply(image_8bit)
    axes[1,i].imshow(clahe_image, cmap='gray')
    axes[1,i].axis('off')

    binarized = np.where(clahe_image > np.mean(clahe_image), 255, 0).astype(np.uint8)
    axes[2,i].imshow(binarized, cmap='gray')
    axes[2,i].axis('off')

    filename = os.path.splitext(os.path.basename(inputs[i]))[0]
    np.save(os.path.join(output, f'{filename}_clahe.npy'), clahe_image)
    np.save(os.path.join(output, f'{filename}_binarized.npy'), binarized)

plt.show()