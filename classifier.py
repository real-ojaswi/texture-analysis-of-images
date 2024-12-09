#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


def load_data(dir='data/lbptraining', image_dir='data/training'):
    X = []  # Features
    y = []  # Labels
    file_names = []  # To keep track of file names

    # Map file prefixes to classes
    class_mapping = {
        'cloudy': 0,
        'rain': 1,
        'shine': 2,
        'sunrise': 3
    }

    # Load saved LBP histograms from the specified directory
    for file_name in os.listdir(dir):
        if file_name.endswith('.npy'):
            # Extract class name from the filename prefix
            class_name = next((cls for prefix, cls in class_mapping.items() if file_name.lower().startswith(prefix)), None)
            if class_name is not None:
                # Load the histogram and append to the features and labels
                histogram = np.load(os.path.join(dir, file_name))
                X.append(histogram)
                y.append(class_name)
                file_names.append(file_name)  # Keep track of the filename

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Load corresponding image data using the filenames
    image_data = []
    for file_name in file_names:
        # Replace the extension with .jpg
        base_file_name = os.path.splitext(file_name)[0] + '.jpg'
        image_path = os.path.join(image_dir, base_file_name)

        # Read the image and resize it if necessary
        image = cv2.imread(image_path)
        if image is not None:
            image_data.append(image)
        else:
            image_path= image_path.replace('jpg', 'jpeg')
            image = cv2.imread(image_path)
            if image is not None:
                image_data.append(image)
            else:
                print(f"Warning: Unable to read image corresponding to {file_name} at {image_path}")
                image_data.append(None)  # Append None if the image could not be loaded

    return X, y, image_data


# In[ ]:


def train_svm_classifier(X_train, y_train, X_test, y_test, degree):
    # Create the SVM classifier
    clf = svm.SVC(kernel='poly', degree=degree)  

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    conf_matrix= confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    return conf_matrix, y_pred


# In[ ]:


def plot_classification_results(y_true, y_pred, image_data, class_names):
    # Create a figure for plotting
    plt.figure(figsize=(15, 10))

    # Create a dictionary to hold correctly classified and misclassified images along with their predicted labels
    classified_images = {cls: {'correct': [], 'incorrect': []} for cls in class_names}

    # Iterate over predictions and store images and predicted labels in the respective lists
    for true_label, predicted_label, img in zip(y_true, y_pred, image_data):
        class_name = class_names[true_label]
        if predicted_label == true_label:
            classified_images[class_name]['correct'].append((img, class_name))  # Store image and predicted label
        else:
            classified_images[class_name]['incorrect'].append((img, class_names[predicted_label]))  # Store image and predicted label

    # Plot results for each class
    for subplot_index, class_name in enumerate(class_names):
        # Plot correctly classified images
        correct_images = classified_images[class_name]['correct']
        incorrect_images = classified_images[class_name]['incorrect']
        
        # Display 1 correctly classified image
        if correct_images:
            plt.subplot(len(class_names), 2, subplot_index * 2 + 1)
            img, predicted_label = correct_images[0]  # Unpack the tuple
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
            plt.title(f'{class_name} (Predicted: {predicted_label})')
            plt.axis('off')

        # Display 1 incorrectly classified image
        if incorrect_images:
            plt.subplot(len(class_names), 2, subplot_index * 2 + 2)
            img, predicted_label = incorrect_images[0]  # Unpack the tuple
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
            plt.title(f'{class_name} (Predicted: {predicted_label})')
            plt.axis('off')

    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['cloudy', 'rain', 'shiny', 'sunrise'],
                yticklabels=['cloudy', 'rain', 'shiny', 'sunrise'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()


# In[ ]:


# laod data
X_train_lbp, y_train_lbp, image_train_lbp = load_data('data/lbptraining', 'data/training')
X_test_lbp, y_test_lbp, image_test_lbp = load_data('data/lbptesting', 'data/testing')


# In[ ]:


# get predictions and confusion matrices
conf_matrix_lbp, y_pred_lbp= train_svm_classifier(X_train_lbp, y_train_lbp, X_test_lbp, y_test_lbp, degree=11)


# In[ ]:


# visualize the results
class_names = ['cloudy', 'rain', 'shiny', 'sunrise'] 
plot_classification_results(y_test_lbp, y_pred_lbp, image_test_lbp, class_names)


# In[ ]:


# plot confusion matrix
plot_confusion_matrix(conf_matrix_lbp)


# In[ ]:


X_train_gm_vgg, y_train_gm_vgg, image_train_gm_vgg = load_data('data/gmtrainingvgg', 'data/training')
X_test_gm_vgg, y_test_gm_vgg, image_test_gm_vgg = load_data('data/gmtestingvgg', 'data/testing')


# In[ ]:


X_train_gm_vgg= X_train_gm_vgg.squeeze(1)


# In[ ]:


X_test_gm_vgg= X_test_gm_vgg.squeeze(1)


# In[ ]:


conf_matrix_gm_vgg, y_pred_gm_vgg= train_svm_classifier(X_train_gm_vgg, y_train_gm_vgg, X_test_gm_vgg, y_test_gm_vgg, degree=2)


# In[ ]:


plot_classification_results(y_test_gm_vgg, y_pred_gm_vgg, image_test_gm_vgg, class_names)


# In[ ]:


plot_confusion_matrix(conf_matrix_gm_vgg)


# In[ ]:


X_train_gm_resnet_fine, y_train_gm_resnet_fine, image_train_gm_resnet_fine = load_data('data/gmtrainingresnet/fine', 'data/training')
X_test_gm_resnet_fine, y_test_gm_resnet_fine, image_test_gm_resnet_fine = load_data('data/gmtestingresnet/fine', 'data/testing')
X_train_gm_resnet_fine= X_train_gm_resnet_fine.squeeze(1)
X_test_gm_resnet_fine= X_test_gm_resnet_fine.squeeze(1)
conf_matrix_gm_resnet_fine, y_pred_gm_resnet_fine= train_svm_classifier(X_train_gm_resnet_fine, y_train_gm_resnet_fine, X_test_gm_resnet_fine, y_test_gm_resnet_fine, degree=3)
plot_classification_results(y_test_gm_resnet_fine, y_pred_gm_resnet_fine, image_test_gm_resnet_fine, class_names)
plot_confusion_matrix(conf_matrix_gm_resnet_fine)


# In[ ]:


X_train_gm_resnet_coarse, y_train_gm_resnet_coarse, image_train_gm_resnet_coarse = load_data('data/gmtrainingresnet/coarse', 'data/training')
X_test_gm_resnet_coarse, y_test_gm_resnet_coarse, image_test_gm_resnet_coarse = load_data('data/gmtestingresnet/coarse', 'data/testing')
X_train_gm_resnet_coarse= X_train_gm_resnet_coarse.squeeze(1)
X_test_gm_resnet_coarse= X_test_gm_resnet_coarse.squeeze(1)
conf_matrix_gm_resnet_coarse, y_pred_gm_resnet_coarse= train_svm_classifier(X_train_gm_resnet_coarse, y_train_gm_resnet_coarse, X_test_gm_resnet_coarse, y_test_gm_resnet_coarse, degree=3)
plot_classification_results(y_test_gm_resnet_coarse, y_pred_gm_resnet_coarse, image_test_gm_resnet_coarse, class_names)
plot_confusion_matrix(conf_matrix_gm_resnet_coarse)


# In[ ]:


X_train_cn, y_train_cn, image_train_cn = load_data('data/cntraining','data/training')
X_test_cn, y_test_cn, image_test_cn = load_data('data/cntesting','data/testing')


# In[ ]:


conf_matrix_cn, y_pred_cn= train_svm_classifier(X_train_cn, y_train_cn, X_test_cn, y_test_cn, degree=1)


# In[ ]:


plot_classification_results(y_test_cn, y_pred_cn, image_test_cn, class_names)


# In[ ]:


plot_confusion_matrix(conf_matrix_cn)


# In[ ]:




