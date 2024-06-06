#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Sat Apr  6 23:47:32 2024
@author: marcalbesa

"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import matplotlib.colors as mcolors

path = "/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/immunos_TFG/DAC_immunos_concentration"

#%% Defining necessary functions

fancy_blue = mcolors.CSS4_COLORS['dodgerblue']
fancy_red = mcolors.CSS4_COLORS['crimson']

def find_duplicate_keys(d, value):
    keys = []
    for k, v in d.items():
        if v == value:
            keys.append(k)  
    if len(keys) > 1:
        result = keys   
    else:
        result = False
    return result

def get_key_from_value(d, value):
    key = None
    for k, v in d.items():
        if v == value:
            key = k
    return key

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

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
    global cluster_to_label_map
    # initialize lists to store iterations
    out_list = []
    labs_list = []
    cluster_to_label_map = {}
    
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
                cluster_counts = []
                for i in range(nb_classes):
                    indices = np.where(preds == i)[0]
                    labs_cluster_x = labs[indices]
                    
                    #store counts vector for similarity
                    label_counts_first = np.array([np.sum(labs_cluster_x == i) for i in range(6)])
                    cluster_counts.append(label_counts_first)
                    
                    #store max value in the cluster
                    mode, freq = np.unique(labs_cluster_x, return_counts=True)
                    max_label = mode[np.argmax(freq)]   
                    cluster_to_label_map[i] = max_label
                    
                first = False
                out_list.append(preds)
            else:
                clusters_mapped = np.zeros_like(preds)
                for i in range(nb_classes):
                    indices = np.where(preds == i)[0]
                    labs_cluster_x = labs[indices]
                    
                    #first option: check if only one cluster with that max
                    mode, freq = np.unique(labs_cluster_x, return_counts=True)
                    max_label_i = mode[np.argmax(freq)]
                    result = find_duplicate_keys(cluster_to_label_map, max_label_i)
                    cluster = get_key_from_value(cluster_to_label_map,max_label_i)
                    if result==False and cluster is not None:
                        clusters_mapped[indices] = cluster
                        
                    #second option: align with highest similarity cluster
                    else:
                        similarities = {}
                        label_counts_i = np.array([np.sum(labs_cluster_x == i) for i in range(6)])
                        check_clusters = range(len(cluster_counts)) if not result else result
                        for max_lab in check_clusters:
                            similarity = cosine_similarity(label_counts_i, cluster_counts[max_lab])
                            similarities[similarity] = max_lab
                        mapped_cluster = similarities[max(similarities.keys())]
                        clusters_mapped[indices] = mapped_cluster
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
    elif nb_classes==8:
        fig, axs = plt.subplots(2, 4, figsize=(12, 7.5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=4
    elif nb_classes==7:
        fig, axs = plt.subplots(2, 4, figsize=(12, 7.5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=4
        axs[1,3].set_visible(False)
    elif nb_classes==6:
        fig, axs = plt.subplots(2, 3, figsize=(10, 7.5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        n=3
    elif nb_classes==5:
        fig, axs = plt.subplots(1, 5, figsize=(12,3))
        plt.subplots_adjust(top=0.75, bottom=0.1)
        n=5
    elif nb_classes==4:
        fig, axs = plt.subplots(1, 4, figsize=(10,3))
        plt.subplots_adjust(top=0.75, bottom=0.1)
        n=5
    elif nb_classes==3:
        fig, axs = plt.subplots(1, 3, figsize=(8,3))
        plt.subplots_adjust(top=0.75, bottom=0.1)
        n=5

    for idx, (cluster_index, cluster_labels) in enumerate(predicted_clusters_labels.items()):
        row = idx // n
        col = idx % n
        bar_positions = range(6)
        label_counts = [np.sum(cluster_labels == i) for i in range(6)]
        if nb_classes >= 6:
            ax = axs[row, col]
        else:
            ax = axs[col]
        
        ax.bar(bar_positions, label_counts, color="blue",alpha = 0.7, width=0.5, align='center')
        highest_freq_column = np.argmax(label_counts)
        ax.bar(bar_positions[highest_freq_column], label_counts[highest_freq_column], color="red", alpha = 0.7,width=0.5, align='center')
        ax.set_title(f"Cluster {cluster_index}")
        ax.set_xlabel('Label')
        ax.set_xticks(range(6))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(False, axis='x')  # Disable vertical grid lines

    fig.suptitle("Histograms of clustering performance across iterations", fontsize=16)  # Add title for the entire plot
    output_path = os.path.join(path, "histograms_clustering_{}clusters.png".format(nb_classes))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    
#%% Computing optimum number of clusters 
min_sim = 0.9

def plot_rms_vs_clusters(sds_inter):
    clusters = list(sds_inter.keys())
    std_devs = list(sds_inter.values())

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.figure(figsize=(8, 4.3), dpi=226), plt.gca()
    ax.plot(clusters, std_devs, marker='o', linestyle='-', color="#1f77b4", linewidth=2, markersize=8, label='RMS')
    ax.set_title('RMS vs. Number of Clusters for Min Similarity among labels of ' + str(min_sim), fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Clusters', fontsize=14)
    ax.set_ylabel('Root Mean Square', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(min(clusters) - 0.1, max(clusters) + 0.1)
    ax.set_ylim(min(std_devs) - 0.03, max(std_devs) + 0.024)
    plt.tight_layout()
    output_path = os.path.join(path, "sd_plots", "RMS_vs_NC_{}.png".format(min_sim))
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    
def find_optimum_clusters(n_min,n_max,min_sim, plot_hists=False, plot_sd_clusters=False):
    global predicted_clusters_labels
    sds_inter = {}
    for nb_classes in range(n_min,n_max+1):
        filtered_counts_list = []
        predicted_clusters_labels = get_frequencies(nb_classes)
        
        if plot_hists:
            plot_histograms(predicted_clusters_labels, nb_classes)
            
        for cluster_labels in predicted_clusters_labels.values():
            label_counts = np.array([np.sum(cluster_labels == i) for i in range(6)])
            filtered_norm_counts = (label_counts[label_counts <=  min_sim * np.max(label_counts)])/np.max(label_counts)
            filtered_counts_list.append(filtered_norm_counts)
        concatenated_array = np.concatenate(filtered_counts_list)
        std_dev = np.sqrt(np.mean(concatenated_array**2))
        sds_inter[nb_classes] = std_dev
        print("For %d clusters,"%(nb_classes),"Standard deviation: ", std_dev)
        
    opt_clusters = min(sds_inter, key=sds_inter.get)
    
    if plot_sd_clusters:
        plot_rms_vs_clusters(sds_inter)
        
    print("For clusters ranging (%d,%d),"%(n_min,n_max),"optimum clusters: ",opt_clusters,", Standard deviation: ", sds_inter[opt_clusters])
    return opt_clusters, sds_inter

# Example usage
opt_clusters, sds_inter = find_optimum_clusters(3, 8, min_sim, plot_hists=True, plot_sd_clusters=True)
