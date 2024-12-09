# **Image Feature Extraction and Classification Using LBP, Gram Matrices, and SVM**

This repository demonstrates how to extract **Local Binary Pattern (LBP)** histograms and **Gram matrices** from images, followed by classification using **Support Vector Machines (SVM)**. The project also utilizes pre-trained models such as **VGG19** and **ResNet** to extract deep features, which are used for texture-based image classification. 

---

## **Features**

1. **RGB to HSI Color Space Conversion**:
   - Converts RGB images to HSI (Hue, Saturation, Intensity) color space to separate chromatic content for enhanced texture and color analysis.

2. **Local Binary Pattern (LBP)**:
   - Computes **LBP histograms** from images to capture texture information.
   - Custom encoding scheme for LBP calculation based on binary interpolation and cyclic rotation.
   - Saves LBP histograms as `.npy` files for further use.

3. **Gram Matrix Extraction Using VGG19 and ResNet**:
   - **Gram matrices** are computed from the feature maps of pre-trained **VGG19** and **ResNet** models.
   - Both fine and coarse levels of Gram matrices are calculated to capture different granularities of feature correlations.
   - Option to compute **channel normalization** descriptors using VGG19 for a more compact representation of features.

4. **SVM Classification**:
   - Uses **SVM classifiers** with polynomial kernels to classify images based on LBP histograms, Gram matrices, and channel normalization descriptors.
   
5. **Visualization**:
   - **Confusion matrices** are plotted to evaluate the classifier’s performance.
   - **Classification results** are visualized with sample images, showing correct and incorrect classifications.

6. **Data Preprocessing and Augmentation**:
   - Supports loading and preprocessing of training and testing data for SVM classification.
   - Handles image resizing and normalization for use with deep learning models.


---


### **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `torch`

