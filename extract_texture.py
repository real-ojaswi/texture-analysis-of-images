#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from skimage.transform import resize
from vgg_and_resnet import VGG19, CustomResNet


# In[ ]:


def rgb_to_hsi(rgb_image):
    '''
    Convert an RGB image to HSI color space.
    Input: 
        rgb_image: 3D array of shape (height, width, 3), with values ranging from 0 to 255.
    Output:
        hsi_image: 3D array of shape (height, width, 3), with H, S, I values.
    '''
    # Convert the RGB image from 0-255 to 0-1 range
    rgb_image = rgb_image / 255.0
    B, G, R = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    
    # Intensity (I)
    I = (R + G + B) / 3.0
    
    # Saturation (S)
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-10)) * min_rgb  # Add a small epsilon to avoid division by zero
    
    # Hue (H)
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-10  # Add epsilon to avoid division by zero
    theta = np.arccos(numerator / denominator)
    
    # Apply conditions for hue calculation
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    
    # Normalize H to the range [0, 1]
    H = H / (2 * np.pi)
    
    # Stack H, S, I into the HSI image
    hsi_image = np.stack([H, S, I], axis=-1)
    
    return hsi_image


# In[ ]:


def get_hue(rgb_image):
    hsi_image= rgb_to_hsi(rgb_image)
    h_channel= hsi_image[:,:,0]
    return h_channel


# In[ ]:


def interpolate(image, x, y):
    """Interpolate the value of the pixel at non-integer coordinates (x, y) using bilinear interpolation."""
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    if x0 + 1 >= image.shape[1] or y0 + 1 >= image.shape[0]:
        return image[y0, x0]  # If out of bounds, use nearest neighbor
    return (image[y0, x0] * (1 - dx) * (1 - dy) +
            image[y0, x0 + 1] * dx * (1 - dy) +
            image[y0 + 1, x0] * (1 - dx) * dy +
            image[y0 + 1, x0 + 1] * dx * dy)


# In[ ]:


def compute_lbp_image(image, radius=1, P=8):
    """Compute the LBP value for each pixel in the image with custom encoding."""
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=int)
    
    # Offsets for neighborhood points
    angles = [2 * np.pi * p / P for p in range(P)]
    offsets = [(radius * np.sin(a), radius * np.cos(a)) for a in angles]

    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center_value = image[i, j]
            binary_values = []

            # Step 1: Binary interpolation and comparison
            for dy, dx in offsets:
                neighbor_value = interpolate(image, j + dx, i + dy)
                binary_values.append(1 if neighbor_value >= center_value else 0)

            # Step 2: Cyclically rotate binary values to minimize zeros
            binary_string = ''.join(map(str, binary_values))
            # print(binary_string)
            min_rotation = min(int(binary_string[p:] + binary_string[:p], 2) for p in range(P))
            binary_string = bin(min_rotation)[2:].zfill(P)
            # print(binary_string)
            

            # Step 3: Encode based on runs of 0s and 1s
            if binary_string.count('1') == P:  # Case: All 1s
                encoding = P
            elif binary_string.count('0') == P:  # Case: All 0s
                encoding = 0
            else:
                zero_runs = zero_runs = [run for run in binary_string.split('1') if run]
                # print(zero_runs)
                # print(len(zero_runs))
                one_runs = zero_runs = [run for run in binary_string.split('0') if run]
                # print(one_runs)
                # print(len(one_runs))
                if len(zero_runs)+len(one_runs) == 2:  # Exactly two runs
                    encoding = binary_string.count('1')
                else:  # More than two runs
                    encoding = P + 1
            # print('\n')
            
            lbp_image[i, j] = encoding
    
    return lbp_image[radius:height-radius,radius:width-radius]


# In[ ]:


def compute_lbp_histogram(image, hue_img=False, lbp_img=False, radius=1, P=8):
    if not hue_img:
        image= get_hue(image)
    if not lbp_img:
        image= compute_lbp_image(image, radius=radius, P=P)
    hist, _ = np.histogram(image, bins=np.arange(P+3), density=True)
    return hist


# In[ ]:


def saveLBP(input_dir= 'data/training', output_dir='data/lbptraining'):
    os.makedirs(output_dir, exist_ok=True)
    image_paths= os.listdir(input_dir)
    full_image_paths= [os.path.join(input_dir, image_path) for image_path in image_paths]
    for i, full_image_path in enumerate(full_image_paths):
        print(f"Processing {i}'th image...")
        image= cv2.imread(full_image_path)
        if image is not None:
            image_hist= compute_lbp_histogram(image)
            
            image_name = os.path.basename(full_image_path)  # Extract the filename
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy")  # Create the output path with .npy extension
            np.save(output_path, image_hist)  # Save the histogram array
        else:
            print(f"Warning: Unable to read image {full_image_path}")


# In[ ]:


def plotLBP(dir='data/lbptraining'):
    # Initialize a dictionary to hold histograms for each class
    histograms = {cls: [] for cls in ['cloudy', 'rain', 'shine', 'sunrise']}

    # Load saved LBP histograms from the specified directory
    for cls in histograms.keys():
        # Check for numpy files in the output directory
        for file_name in os.listdir(dir):
            if file_name.lower().startswith(cls) and file_name.endswith('.npy'):
                # Load the histogram and append it to the appropriate class
                histogram = np.load(os.path.join(dir, file_name))
                histograms[cls].append(histogram)
            

    # Plot histograms
    plt.figure(figsize=(15, 9)) 

    # Maximum number of images to display for each class (3)
    max_images = 3

    for subplot_index, cls in enumerate(histograms.keys()):
        class_histograms = histograms[cls]
        for i in range(max_images):
            plt.subplot(max_images, len(histograms), i * len(histograms)  + subplot_index+ 1)  # Correct indexing

            if i < len(class_histograms):
                plt.bar(np.arange(len(class_histograms[i])), class_histograms[i], width=0.5, align='center')
                plt.xticks(np.arange(len(class_histograms[i])), np.arange(len(class_histograms[i])), rotation=45)
                plt.ylabel('Frequency')
            else:
                plt.axis('off')  # Turn off the axis for empty subplots

            # Set the title only for the first row
            if i == 0:
                plt.title(f"{cls.capitalize()}", fontsize=12)

            # Common Y-axis for all images in the same column
            if subplot_index == 0:
                plt.ylabel('Frequency')
            
            # Set common X-axis only for the first row
#             if i == 0:
#                 plt.xlabel('LBP Value')

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the histograms


# In[ ]:


def saveGMvgg(model, input_dir='data/training', output_dir='data/gmtrainingvgg', resized=True, save_channel_norm=False):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = os.listdir(input_dir)
    full_image_paths = [os.path.join(input_dir, image_path) for image_path in image_paths]
    
    for i, full_image_path in enumerate(full_image_paths):
        print(f"Processing {i}'th image...")
        image = cv2.imread(full_image_path)
        
        if image is not None:
            # Resize the image as required by the model
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            F = model(image)  # Obtain the feature map from the model

            
            if save_channel_norm:
                # Initialize lists to hold mean and variance
                means = []
                variances = []
                Nl, height, width = F.shape  # Unpack the shape of F
                
                for i in range(Nl):
                    # Flatten the feature map for channel i to calculate mean and variance
                    feature_channel = F[i, :, :].flatten()  # Shape will be (14*14,)
                    
                    # Calculate mean for channel i
                    mean = np.mean(feature_channel)
                    means.append(mean)
                    
                    # Calculate variance for channel i
                    variance = np.var(feature_channel)
                    variances.append(variance)
                
                # Concatenate means and variances
                vnorm = np.concatenate([means, variances])
#                 print(f"Channel normalization descriptor shape: {vnorm.shape}")
                
                # Save the descriptor
                image_name = os.path.basename(full_image_path)  # Extract the filename
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy")
                np.save(output_path, vnorm)  # Save the normalization descriptor
            
            else:
                # Compute Gram matrix
                F = F.reshape(512, -1)  # Reshape to (512, 196) where 196 = 14*14
                G = np.matmul(F, F.T)  # Compute Gram matrix
                G = G / np.max(G)  # Normalize the Gram matrix
                if resized:
                    G = resize(G, (32, 32), mode='reflect', anti_aliasing=True) # Downsample the gram matrix
                    G = G[np.triu_indices(32)] # Extract the upper triangular part 
                G = G.reshape(1, -1)  # Reshape for saving
                
                # Save the Gram matrix
                image_name = os.path.basename(full_image_path)  # Extract the filename
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy")
                np.save(output_path, G)  # Save the Gram matrix
            
        else:
            print(f"Warning: Unable to read image {full_image_path}")


# In[ ]:


def saveGMresnet(model, input_dir='data/training', output_dir='data/gmtrainingresnet', resized=True):
    output_dir_fine= os.path.join(output_dir, 'fine')
    output_dir_coarse= os.path.join(output_dir, 'coarse')
    os.makedirs(output_dir_fine, exist_ok=True)
    os.makedirs(output_dir_coarse, exist_ok=True)
    image_paths = os.listdir(input_dir)
    full_image_paths = [os.path.join(input_dir, image_path) for image_path in image_paths]
    
    for i, full_image_path in enumerate(full_image_paths):
        print(f"Processing {i}'th image...")
        image = cv2.imread(full_image_path)
        
        if image is not None:
            # Resize the image as required by the model
            image = resize(image, (512, 512), anti_aliasing=True)
            F_fine, F_coarse = model(image)  # Obtain the feature map from the model
            # Save the descriptor
            image_name = os.path.basename(full_image_path)  # Extract the filename
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy")
            # Compute Gram matrix
            F_fine = F_fine.reshape(512, -1)  # Reshape to (512, 196) where 196 = 14*14
            G_fine = np.matmul(F_fine, F_fine.T)  # Compute Gram matrix
            G_fine = G_fine / np.max(G_fine)  # Normalize the Gram matrix
            if resized:
                G_fine = resize(G_fine, (32, 32), mode='reflect', anti_aliasing=True) # Downsample the gram matrix
                G_fine = G_fine[np.triu_indices(32)] # Extract the upper triangular part 
            G_fine = G_fine.reshape(1, -1)  # Reshape for saving
            
            F_coarse = F_coarse.reshape(512, -1)  # Reshape to (512, 196) where 196 = 14*14
            G_coarse = np.matmul(F_coarse, F_coarse.T)  # Compute Gram matrix
            G_coarse = G_coarse / np.max(G_coarse)  # Normalize the Gram matrix
            if resize:
                G_coarse = resize(G_coarse, (32, 32), mode='reflect', anti_aliasing=True) # Downsample the gram matrix
                G_coarse = G_coarse[np.triu_indices(32)] # Extract the upper triangular part 
            G_coarse = G_coarse.reshape(1, -1)  # Reshape for saving

            # Save the Gram matrix
            image_name = os.path.basename(full_image_path)  # Extract the filename
            output_path_fine = os.path.join(output_dir_fine, f"{os.path.splitext(image_name)[0]}.npy")
            np.save(output_path_fine, G_fine)  # Save the Gram matrix from fine features

            output_path_coarse = os.path.join(output_dir_coarse, f"{os.path.splitext(image_name)[0]}.npy")
            np.save(output_path_coarse, G_coarse)  # Save the Gram matrix from fine features
        else:
            print(f"Warning: Unable to read image {full_image_path}")


# In[ ]:


def plotGM(dir='data/gmtraining'):
    # Initialize a dictionary to hold Gram matrices for each class
    gram_matrices = {cls: [] for cls in ['cloudy', 'rain', 'shine', 'sunrise']}

    for cls in gram_matrices.keys():
        # Check for numpy files in the output directory
        for file_name in os.listdir(dir):
            if file_name.lower().startswith(cls) and file_name.endswith('.npy'):
                # Load the Gram matrix, reshape it to 512x512, normalize it to [0, 255], and append to the appropriate class
                gm_matrix = np.load(os.path.join(dir, file_name))
                gm_matrix = gm_matrix.reshape(512, 512)  # Reshape to 512x512
                
                # Normalize the Gram matrix to the range [0, 255]
                gm_matrix = (gm_matrix - gm_matrix.min()) / (gm_matrix.max() - gm_matrix.min()) * 255
                gm_matrix = gm_matrix.astype(np.uint8)  # Convert to unsigned 8-bit integer

                gram_matrices[cls].append(gm_matrix)

    # Plot Gram matrices
    plt.figure(figsize=(15, 9)) 

    # Maximum number of images to display for each class (3)
    max_images = 3

    for subplot_index, cls in enumerate(gram_matrices.keys()):
        class_matrices = gram_matrices[cls]
        for i in range(max_images):
            plt.subplot(max_images, len(gram_matrices), i * len(gram_matrices) + subplot_index + 1)  # Correct indexing

            if i < len(class_matrices):
                plt.imshow(class_matrices[i], cmap='plasma', interpolation='nearest')
                plt.axis('off')  # Turn off the axis
            else:
                plt.axis('off')  # Turn off the axis for empty subplots

            # Set the title only for the first row
            if i == 0:
                plt.title(f"{cls.capitalize()}", fontsize=12)

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the Gram matrices


# In[ ]:


#load VGG19
vgg19= VGG19()
vgg19.load_weights('/scratch/gilbreth/oachary/ECE66100/vgg_normalized.pth')


# In[ ]:


# load resnet
resnet= CustomResNet(encoder='resnet50')


# In[ ]:


# save gram matrices obtained using resnet backbone
saveGMresnet(resnet)
saveGMresnet(resnet, 'data/testing', 'data/gmtestingresnet')


# In[ ]:


# visualize the gram matrices obtained using resnet backbone
plotGM('data/gmtrainingresnet/fine')


# In[ ]:


plotGM('data/gmtrainingresnet/coarse')


# In[ ]:


# save gram matrices obtained using vgg19 backbone
saveGMvgg(vgg19)
saveGMvgg(vgg19, input_dir='data/testing', output_dir='data/gmtestingvgg', save_channel_norm=False)


# In[ ]:


# save channel normalization based descriptor obtained using vgg19 backbone
saveGMvgg(vgg19, input_dir='data/training', output_dir='data/cntraining', save_channel_norm=True)
saveGMvgg(vgg19, input_dir='data/testing', output_dir='data/cntesting', save_channel_norm=True)


# In[ ]:


# visualize gram matrices obtained using vgg19 backbone
# plotGM('data/gmtrainingresnet/fine')


# In[ ]:


# save LBP
saveLBP()
saveLBP('data/testing', 'data/lbptesting')


# In[ ]:


# visualize LBP
plotLBP()

