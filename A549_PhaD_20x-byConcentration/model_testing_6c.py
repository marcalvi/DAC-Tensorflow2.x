#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Sun May 19 21:24:13 2024
@author: marcalbesa

"""

#%% Import packages and configuration

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Activation, Flatten, Input, MaxPooling2D, BatchNormalization
from keras.src import backend
from skimage.transform import resize
from numpy.linalg import norm
from PIL import Image
from collections import defaultdict
from scipy.interpolate import interp1d
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit

class MyBatchNorm(BatchNormalization):

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
            # if not self.trainable:
            #     # When the layer is not trainable, it overrides the value passed
            #     # from model.
            #     training = False
        return training

#%% Creating the ConvNet

def ConvNetwork(input_shape, nb_classes, trainBN, mom):
    # First convolutional block
    inp_img = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(inp_img)
    bn1 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu1)
    bn2 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv2)
    relu2 = Activation('relu')(bn2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu2)
    bn3 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv3)
    relu3 = Activation('relu')(bn3)

    # First max pool block
    max1 = MaxPooling2D((2, 2), (2, 2),padding='same')(relu3)

    # Second convolutional block
    conv4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max1)
    bn5 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv4)
    relu4 = Activation('relu')(bn5)
    conv5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu4)
    bn6 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv5)
    relu5 = Activation('relu')(bn6)
    conv6 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu5)
    bn7 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv6)
    relu6 = Activation('relu')(bn7)

    # Second max pool block
    max2 = MaxPooling2D((2, 2), (2, 2), padding='same')(relu6)

    # Classifier block
    conv7 = Conv2D(10, (1, 1), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max2)
    bn9 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(conv7)
    relu7 = Activation('relu')(bn9)

    # Average Pooling
    avg = AveragePooling2D((2, 2), (2, 2), padding='same')(relu7)

    # First dense block
    flat = Flatten()(avg)
    dense1 = Dense(nb_classes, kernel_initializer= 'identity')(flat)
    bn11 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(dense1)
    relu8 = Activation('relu')(bn11)

    # Second dense block
    dense2 = Dense(nb_classes, kernel_initializer= 'identity')(relu8)
    bn12 = MyBatchNorm(momentum = mom, epsilon=1e-5, trainable = trainBN)(dense2)
    relu8 = Activation('relu')(bn12)
    out = Activation('softmax')(relu8)

    cluster_l1 = Model(inputs=[inp_img], outputs=[out])

    return cluster_l1

#%% Defining the functions

# Function to ensure unique values in label_to_cluster
def ensure_unique_values(d):
    unique_values = set(d.values())
    return len(unique_values) == len(d.values())

# function to preprocess the MNIST dataset
def preprocess_data(X_data,mode, normalize= False):
    X_data = X_data/np.max(X_data)
    return X_data

def get_key_from_value(d, value):
    keys = []
    for k, v in d.items():
        if v == value:
            keys.append(k)  
    return keys[0]

def remove_duplicates_ordered(array):
    array = np.array(array)
    unique_values, index = np.unique(array, return_index=True)
    return array[np.sort(index)]

# Filtering for confidence
def filter_preds(pred_cluster):
    top_two_indices = np.argsort(np.sum(pred_cluster, axis=0))[::-1][:2]  
    mask = np.ones(pred_cluster.shape[1], dtype=bool)
    mask[top_two_indices] = False  
    pred_cluster[:, mask] = 0  
    max_columns = np.argmax(pred_cluster[:, top_two_indices], axis=1)
    mask_rows = np.any(max_columns == 0, axis=0) | np.any(max_columns == 1, axis=0) 
    filtered_preds = pred_cluster[mask_rows][0,:,:]
    return filtered_preds

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def sigmoid(x, x0, k, ymax, ymin):
    return ymin + (ymax - ymin) / (1 + np.exp(-k*(x-x0)))

#%% Importing the dataset

# Images path
conditions_list = ["Control", "1Fe", "5Fe", "10Fe", "50Fe", "100Fe", "1Ti", "5Ti",
                   "10Ti", "50Ti", "100Ti", "1Sn", "5Sn", "10Sn", "50Sn", "100Sn"]
files_path = "/Users/marcalbesa/Desktop/TFG/fotos_immunos_norm"

# List to store the loaded images
images_control = []
images_Fe_1 = []
images_Fe_5 = []
images_Fe_10 = []
images_Fe_50 = []
images_Fe_100 = []
images_Ti_1 = []
images_Ti_5 = []
images_Ti_10 = []
images_Ti_50 = []
images_Ti_100 = []
images_Sn_1 = []
images_Sn_5 = []
images_Sn_10 = []
images_Sn_50 = []
images_Sn_100 = []
images_list = [images_control, images_Fe_1, images_Fe_5, images_Fe_10, images_Fe_50, images_Fe_100, images_Ti_1, images_Ti_5,
                images_Ti_10, images_Ti_50, images_Ti_100, images_Sn_1, images_Sn_5, images_Sn_10, images_Sn_50, images_Sn_100]

# Directory containing TIFF images
for i, cond in enumerate(conditions_list):
        path = os.path.join(files_path, "ML_A549_"+cond+"_PhaD_20X_RGB")
        for filename in os.listdir(path):
            if filename.endswith(".tif"):
                file_path = os.path.join(path, filename)
                tiff_image = np.array(Image.open(file_path))
                image = resize(tiff_image, (256, 256,3))
                images_list[i].append(image)

# Creating test dataset for materials and concentration
X_test_C = np.concatenate(images_list, axis=0)
X_test = preprocess_data(X_test_C, "training", normalize=False)

# Creating test labels
yC = np.zeros(len(images_control))
y1Fe = 1 * np.ones(len(images_Fe_1))
y5Fe = 2 * np.ones(len(images_Fe_5))
y10Fe = 3 * np.ones(len(images_Fe_10))
y50Fe = 4 * np.ones(len(images_Fe_50))
y100Fe = 5 * np.ones(len(images_Fe_100))
y1Ti = 6 * np.ones(len(images_Ti_1))
y5Ti = 7 * np.ones(len(images_Ti_5))
y10Ti = 8 * np.ones(len(images_Ti_10))
y50Ti = 9 * np.ones(len(images_Ti_50))
y100Ti = 10 * np.ones(len(images_Ti_100))
y1Sn = 11 * np.ones(len(images_Sn_1))
y5Sn = 12 * np.ones(len(images_Sn_5))
y10Sn = 13 * np.ones(len(images_Sn_10))
y50Sn = 14 * np.ones(len(images_Sn_50))
y100Sn = 15 * np.ones(len(images_Sn_100))
y_test = np.hstack((yC, y1Fe, y5Fe, y10Fe, y50Fe, y100Fe, y1Ti, y5Ti, y10Ti, y50Ti,
                    y100Ti, y1Sn, y5Sn, y10Sn, y50Sn, y100Sn)).astype(int)

#%% Reloading model weights

nb_classes = 6
weights_path = "/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/immunos_TFG/DAC_immunos_concentration"

# define input shape and thresholds
img_rows, img_cols, img_channels = 256, 256, 3
inp_shape = (img_rows, img_cols, img_channels)

# Reload the model and load weights
model = ConvNetwork(inp_shape, nb_classes = nb_classes, trainBN = True, mom = 0.8)
model.load_weights(os.path.join(weights_path,"{}_clusters/DAC_weights/DAC_weights_{}clusters_iter6.h5".format(nb_classes,nb_classes)))
# model.load_weights(os.path.join(weights_path,"{}_clusters/DAC_weights/DAC_weights_{}clusters_bo.h5".format(nb_classes,nb_classes)))

#%% Mapping results

label_to_cluster = {}
true_labels = [0, 1, 10, 50, 80, 100]  # Replace with Transwell results
Ti_locs = [0] + list(range(6, 11))

label_to_damage = {}
for i in range(6):
    label_to_damage[Ti_locs[i]] = true_labels[i]

label_to_cluster = {}
cluster_to_count = defaultdict(int)

# Iterate through Ti_locs to determine the highest frequency clusters
for i in Ti_locs:  # labels corresponding to Ti
    indices = np.where(y_test == i)[0]
    images = X_test[indices]
    preds = model.predict(images, verbose=1, batch_size=16)
    pred_cluster = np.argmax(preds, axis=1)
    print(pred_cluster)
    cluster_counts = [np.sum(pred_cluster == j) for j in range(nb_classes)]
    
    # Find the clusters sorted by frequency in descending order
    sorted_clusters = sorted(range(nb_classes), key=lambda k: cluster_counts[k], reverse=True)
    
    # Assign the most frequent cluster to the label
    for cluster in sorted_clusters:
        if cluster not in label_to_cluster.values():
            label_to_cluster[i] = cluster
            cluster_to_count[cluster] = cluster_counts[cluster]
            break
        else:
            existing_label = get_key_from_value(label_to_cluster, cluster)
            existing_count = cluster_to_count[cluster]
            
            # Compare frequencies and update if necessary
            if cluster_counts[cluster] > existing_count:
                label_to_cluster[i] = cluster
                cluster_to_count[cluster] = cluster_counts[cluster]
                label_to_cluster[existing_label] = None  # Mark the old label for reassignment
                break

# Reassign any labels marked for reassignment
for i in Ti_locs:
    if label_to_cluster[i] is None:
        indices = np.where(y_test == i)[0]
        images = X_test[indices]
        preds = model.predict(images, verbose=1, batch_size=16)
        pred_cluster = np.argmax(preds, axis=1)
        cluster_counts = [np.sum(pred_cluster == j) for j in range(nb_classes)]
        
        # Find the next most frequent cluster not already assigned
        sorted_clusters = sorted(range(nb_classes), key=lambda k: cluster_counts[k], reverse=True)
        for cluster in sorted_clusters:
            if cluster not in label_to_cluster.values():
                label_to_cluster[i] = cluster
                cluster_to_count[cluster] = cluster_counts[cluster]
                break

sorted_damage_labels = []
for i in range(nb_classes):
    labs = get_key_from_value(label_to_cluster, i)
    sorted_damage_labels.append(label_to_damage[labs])

print("Sorted damage labels to match cluster order: ",sorted_damage_labels)

#%% Regressing predictions to damage labels
# yC = 0   y1Fe = 1    y5Fe = 2    y10Fe = 3    y50Fe = 4    y100Fe = 5
#          y1Ti = 6    y5Ti = 7    y10Ti = 8    y50Ti = 9    y100Ti = 10
#          y1Sn = 11   y5Sn = 12   y10Sn = 13   y50Sn = 14   y100Sn = 15

# Sorted damage labels to match cluster order: [25, 100, 50, 0]

# Find images with label
indices = np.where(y_test ==10)[0]
imatges = X_test[indices]

# Get model predictions
pred_cluster = model.predict(imatges)
filtered_preds = filter_preds(pred_cluster)

# Compute the scalar product to assess output label
sorted_damage_labels = remove_duplicates_ordered(sorted_damage_labels)
scalar_product = np.dot(filtered_preds, sorted_damage_labels)

print("Thresholded model output: \n",filtered_preds)
print("Predicted barrier damage: " + str(round(np.mean(scalar_product),2))+"%")

#%% Plotting the boxplot

# Creating labels for the boxplot
class_labels = ['Control','1 µg/ml','5 µg/ml','10 µg/ml','50 µg/ml','100 µg/ml']

# Creating necessary lists
Ti_locs = [0] + list(range(6,11))
predictions_Ti = []
sorted_damage_labels = remove_duplicates_ordered(sorted_damage_labels)

# Training predictions
ref_preds = []
for i in range(len(Ti_locs)):
    indices = np.where(y_test == Ti_locs[i])[0]
    imatges = X_test[indices]
    preds = model.predict(imatges, verbose=1, batch_size=16)
    ref_preds.append(np.mean(preds, axis=0))
    filtered_preds = filter_preds(preds)
    scalar_product = np.dot(filtered_preds, sorted_damage_labels)
    predictions_Ti.append(scalar_product)
    

#%%
# Test predictions
Fe_locs = [0] + list(range(1,6))
Sn_locs = [0] + list(range(11,16))
locs = [Fe_locs,Sn_locs]
predictions_Fe = []
predictions_Sn= []
predictions = [predictions_Fe,predictions_Sn]
for m in range(len(locs)):
    for conc in range(len(locs[m])): 
        indices = np.where(y_test == locs[m][conc])[0]
        imatges = X_test[indices]
        preds = model.predict(imatges, verbose=1, batch_size=16)
        filtered_preds = filter_preds(preds)
        similar_preds = []
        for j in range(len(filtered_preds)):
            cos_sim = cosine_similarity(preds[j], ref_preds[conc]) 
            if cos_sim >= 0.9:  # Check if the similarity is at least 50%
                similar_preds.append(filtered_preds[j])
        if similar_preds:
            scalar_product = np.dot(np.array(similar_preds), sorted_damage_labels)
            predictions[m].append(scalar_product)
        else:
            for j in range(len(filtered_preds)):
                cos_sim = cosine_similarity(preds[j], ref_preds[conc]) 
                if cos_sim >= 0.01: 
                    similar_preds.append(filtered_preds[j])
            scalar_product = np.dot(np.array(similar_preds), sorted_damage_labels)
            predictions[m].append(scalar_product)

#%% Plotting

fig, ax = plt.subplots(figsize=(10, 6))
boxplot = ax.boxplot(predictions_Fe,labels = class_labels)
mean_values = [line.get_ydata()[0] for line in boxplot['medians']]
positions = [line.get_xdata()[0]+0.25 for line in boxplot['medians']]
f = interp1d(positions, mean_values, kind='quadratic')
new_positions_fe = np.linspace(min(positions), max(positions), 100)
smoothed_upper_quartiles_fe = f(new_positions_fe)
ax.plot(new_positions_fe, smoothed_upper_quartiles_fe, color='blue')
plt.title('Boxplot of Predictions for FeO Particles for {} clusters'.format(nb_classes))
plt.xlabel('Class')
plt.ylabel('Predictions [%]')
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
boxplot = ax.boxplot(predictions_Ti,labels = class_labels)
mean_values = [line.get_ydata()[0] for line in boxplot['medians']]
positions = [line.get_xdata()[0]+0.25 for line in boxplot['medians']]
f = interp1d(positions, mean_values, kind='quadratic')
new_positions_ti = np.linspace(min(positions), max(positions), 100)
smoothed_upper_quartiles_ti = f(new_positions_ti)
ax.plot(new_positions_ti, smoothed_upper_quartiles_ti, color='blue')
plt.title('Boxplot of Predictions for TiO Particles for {} clusters'.format(nb_classes))
plt.xlabel('Class')
plt.ylabel('Predictions [%]')
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
boxplot = ax.boxplot(predictions_Sn,labels = class_labels)
mean_values = [line.get_ydata()[0] for line in boxplot['medians']]
positions = [line.get_xdata()[0]+0.25 for line in boxplot['medians']]
f = interp1d(positions, mean_values, kind='quadratic')
new_positions_sn = np.linspace(min(positions), max(positions), 100)
smoothed_upper_quartiles_sn = f(new_positions_sn)
ax.plot(new_positions_sn, smoothed_upper_quartiles_sn, color='blue')
plt.title('Boxplot of Predictions for Sn Particles for {} clusters'.format(nb_classes))
plt.xlabel('Class')
plt.ylabel('Predictions [%]')
plt.grid(True)
plt.show()

#%% Violin plot

path = "/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/immunos_TFG/DAC_immunos_concentration/violin_plots"

# TiO plotting
data = {'Category': [],'Value': []}
for category, value_list in zip(class_labels, predictions_Ti):
    data['Category'].extend([category] * len(value_list))
    data['Value'].extend(value_list)
df = pd.DataFrame(data)
    
medians = []
for val in predictions_Ti:
    medians.append(np.median(val))
p0 = [np.median(medians), 1, max(medians), min(medians)]  # Initial guess for parameters
params, _ = curve_fit(sigmoid, np.arange(len(class_labels)), medians)
x_values = np.linspace(0, len(class_labels)-1, 100)
    
# Create a violin plot with a swarm plot overlay
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", palette="muted")
violin = sns.violinplot(x='Category', y='Value', data=df, inner=None, linewidth=1, palette='flare',hue = df["Category"])
swarm = sns.stripplot(x='Category', y='Value', data=df, color='k', alpha=0.6, size=4)
plt.plot(x_values, sigmoid(x_values, *params), color='crimson', label='Sigmoid Curve',linewidth=2)
plt.title('Predictions for A549 exposed to TiO2 Particles for {} clusters'.format(nb_classes))
plt.xlabel('Concentration')
plt.ylim([0,100])
plt.ylabel('Loss of Function (%)')
plt.savefig(os.path.join(path,"violin_Ti_{}clusters.png".format(nb_classes)))
plt.show()

# FeO plotting
data = {'Category': [],'Value': []}
for category, value_list in zip(class_labels, predictions_Fe):
    data['Category'].extend([category] * len(value_list))
    data['Value'].extend(value_list)
df = pd.DataFrame(data)
    
medians = []
for val in predictions_Fe:
    medians.append(np.median(val))
p0 = [np.median(medians), 1, max(medians), min(medians)]  # Initial guess for parameters
params, _ = curve_fit(sigmoid, np.arange(len(class_labels)), medians, maxfev=20000)
x_values = np.linspace(0, len(class_labels)-1, 100)
    
# Create a violin plot with a swarm plot overlay
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", palette="muted")
violin = sns.violinplot(x='Category', y='Value', data=df, inner=None, linewidth=1, palette='flare',hue = df["Category"])
swarm = sns.stripplot(x='Category', y='Value', data=df, color='k', alpha=0.6, size=4)
plt.plot(x_values, smoothed_upper_quartiles_fe, color='crimson')
plt.title('Predictions for A549 exposed to Fe2O3 Particles for {} clusters'.format(nb_classes))
plt.xlabel('Concentration')
plt.ylim([0,100])
plt.ylabel('Loss of Function (%)')
plt.savefig(os.path.join(path,"violin_Fe_{}clusters.png".format(nb_classes)))
plt.show()

# Sn plotting
data = {'Category': [],'Value': []}
for category, value_list in zip(class_labels, predictions_Sn):
    data['Category'].extend([category] * len(value_list))
    data['Value'].extend(value_list)
df = pd.DataFrame(data)
    
medians = []
for val in predictions_Sn:
    medians.append(np.median(val))
p0 = [np.median(medians), 1, max(medians), min(medians)]  # Initial guess for parameters
params, _ = curve_fit(sigmoid, np.arange(len(class_labels)), medians)
x_values = np.linspace(0, len(class_labels)-1, 100)
    
# Create a violin plot with a swarm plot overlay
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", palette="muted")
violin = sns.violinplot(x='Category', y='Value', data=df, inner=None, linewidth=1, palette='flare',hue = df["Category"])
swarm = sns.stripplot(x='Category', y='Value', data=df, color='k', alpha=0.6, size=4)
plt.plot(x_values, smoothed_upper_quartiles_sn, color='crimson')
plt.title('Predictions for A549 exposed to Sn Particles for {} clusters'.format(nb_classes))
plt.xlabel('Concentration')
plt.ylim([0,100])
plt.ylabel('Loss of Function (%)')
plt.savefig(os.path.join(path,"violin_Sn_{}clusters.png".format(nb_classes)))
plt.show()
