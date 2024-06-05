#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Sat Apr  6 23:47:32 2024
@author: marcalbesa

"""
import os, h5py
import numpy as np
import matplotlib.pyplot as plt

nb_classes = 4
path = "/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/immunos_TFG/DAC_immnos_material"

#%% Getting files
def get_files(nb_classes):
    global path
    path_clusters = os.path.join(path, "%d_clusters" %nb_classes)
    path_out_folder = os.path.join(path_clusters,"DAC_outputs")
    
    # append filenames containing .h5
    file_list = []
    for filename in os.listdir(path_out_folder):
        if filename.endswith('.h5'):  # Check if the file is an HDF5 file
            file_list.append(os.path.join(path_out_folder, filename))
    return file_list

#%% Getting frequencies

def get_frequencies(nb_classes):
    # initialize lists to store iterations
    out_list = []
    labs_list = []
    label_to_cluster_map = {}
    
    filepaths_list = get_files(nb_classes)
    
    first = True
    # iterate over the list of filenames
    for path_out_file in filepaths_list:
        with h5py.File(path_out_file, 'r') as hf:
            out = hf['/outputs'][:]
            labs = hf['/labels'][:]
            
            # find predicted cluster, concatenate and append to the lists
            preds = np.argmax(out, axis=-1)
            preds = np.concatenate(preds, axis=0)
            labs = np.concatenate(labs, axis=0)
            
            # aligning clusters between iterations
            if first:
                for i in range(nb_classes):
                    indices = np.where(preds == i)[0]
                    labs_cluster_x = labs[indices]
                    mode, freq = np.unique(labs_cluster_x, return_counts=True)
                    label_from_cluster = mode[np.argmax(freq)]   
                    label_to_cluster_map[label_from_cluster] = i
                    freq_max = freq[np.argmax(freq)]
                    freq[np.argmax(freq)] = 0
                    freq_2max = freq[np.argmax(freq)]
                    th = 0.7 if nb_classes>6 else 0.99
                    if freq_2max/freq_max >= th:
                        label_from_cluster = mode[np.argmax(freq)]
                        label_to_cluster_map[label_from_cluster] = i
                first = False
                out_list.append(preds)
            else:
                clusters_mapped = np.zeros_like(preds)
                for i in range(nb_classes):
                    indices = np.where(preds == i)[0]
                    labs_cluster_x = labs[indices]
                    mode, freq = np.unique(labs_cluster_x, return_counts=True)
                    label_from_cluster = mode[np.argmax(freq)]                   
                    try:
                        clusters_mapped[indices] = label_to_cluster_map[label_from_cluster]
                    except KeyError:
                        freq[np.argmax(freq)] = 0
                        label_from_cluster = mode[np.argmax(freq)]  
                        clusters_mapped[indices] = label_to_cluster_map[label_from_cluster]
                out_list.append(clusters_mapped)
            labs_list.append(labs)

    predictions_stack = np.concatenate(out_list, axis=0)
    labels_stack = np.concatenate(labs_list, axis=0)

    # initialize the dictionary and store the vectors
    predicted_clusters_labels = {}
    for n in range(nb_classes):
        cluster_indices = np.where(predictions_stack == n)[0]
        cluster_labels = labels_stack[cluster_indices]
        predicted_clusters_labels[n] = cluster_labels
        
    return predicted_clusters_labels

#%% Visualization

def plot_histograms(predicted_clusters_labels, nb_classes):
    if nb_classes==10:
        fig, axs = plt.subplots(2, 5, figsize=(12,6))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=5
    elif nb_classes==9:
        fig, axs = plt.subplots(3, 3, figsize=(7.5,9))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=3
    elif nb_classes==8 or nb_classes==7:
        fig, axs = plt.subplots(2, 4, figsize=(12, 7.5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=4
    elif nb_classes==6:
        fig, axs = plt.subplots(2, 3, figsize=(10, 7.5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=3
    elif nb_classes==5:
        fig, axs = plt.subplots(1, 5, figsize=(12,2.5))
        plt.subplots_adjust(top=0.75, bottom=0.1, hspace=0.4)
        n=5
    elif nb_classes==4:
        fig, axs = plt.subplots(1, 4, figsize=(10,2.5))
        plt.subplots_adjust(top=0.75, bottom=0.1, hspace=0.4)
        n=5
    elif nb_classes==3:
        fig, axs = plt.subplots(1, 3, figsize=(8,2.5))
        plt.subplots_adjust(top=0.75, bottom=0.1, hspace=0.4)
        n=5

    
    for idx, (cluster_index, cluster_labels) in enumerate(predicted_clusters_labels.items()):
        row = idx // n
        col = idx % n
        bar_positions = range(4)
        if nb_classes >=6:
            label_counts = [np.sum(cluster_labels == i) for i in range(4)]
            axs[row, col].bar(bar_positions, label_counts, color='blue', alpha=0.7, width=0.5, align='center')
            highest_freq_column = np.argmax(label_counts)
            axs[row, col].bar(bar_positions[highest_freq_column], label_counts[highest_freq_column], color='red', alpha=0.7, width=0.5, align='center')
            axs[row, col].set_title(f"Cluster {cluster_index}")
            axs[row, col].set_xlabel('Label')
            axs[row, col].set_xticks(range(4))
            axs[row, col].set_yticks([])
            axs[row, col].set_yticklabels([])
        else:
            label_counts = [np.sum(cluster_labels == i) for i in range(4)]
            axs[col].bar(bar_positions, label_counts, color='blue', alpha=0.7, width=0.5, align='center')
            highest_freq_column = np.argmax(label_counts)
            axs[col].bar(bar_positions[highest_freq_column], label_counts[highest_freq_column], color='red', alpha=0.7, width=0.5, align='center')
            axs[col].set_title(f"Cluster {cluster_index}")
            axs[col].set_xlabel('Label')
            axs[col].set_xticks(range(4))
            # axs[col].set_xticks(["Control","FeO","TiO","Sn"])
            axs[col].set_yticks([])
            axs[col].set_yticklabels([])
    
    fig.suptitle("Histograms of clustering performance across iterations", fontsize=16)  # Add title for the entire plot
    plt.savefig(os.path.join(path,"histograms_clustering.png"))
    plt.show()

#%% Computing optimum number of clusters 

def find_optimum_clusters(n_min,n_max):
    global predicted_clusters_labels, std_dev,cluster_labels,label_counts
    sds_inter = {}
    for nb_classes in range(n_min,n_max+1):
        filtered_counts_list = []
        predicted_clusters_labels = get_frequencies(nb_classes)
        plot_histograms(predicted_clusters_labels, nb_classes)
        for cluster_labels in predicted_clusters_labels.values():
            label_counts = np.array([np.sum(cluster_labels == i) for i in range(4)])
            norm_factor = np.max(label_counts)
            filtered_norm_counts = (label_counts[label_counts <= 0.6* np.max(label_counts)])/norm_factor
            filtered_counts_list.append(filtered_norm_counts)
        concatenated_array = np.concatenate(filtered_counts_list)
        std_dev = np.std(concatenated_array)
        sds_inter[nb_classes] = std_dev
        print("For %d clusters,"%(nb_classes),"Standard deviation: ", std_dev)
    opt_clusters = min(sds_inter, key=sds_inter.get)
    print("For clusters ranging (%d,%d),"%(n_min,n_max),"optimum clusters: ",opt_clusters,", Standard deviation: ", sds_inter[opt_clusters])
    return opt_clusters
    
opt_clusters = find_optimum_clusters(3,5)

