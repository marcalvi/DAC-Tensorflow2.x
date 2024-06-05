#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Tue May 21 03:48:38 2024
@author: marcalbesa

"""

#%%

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_classes", type=int)
    parser.add_argument("--mom", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=int)

    args = parser.parse_args()
    nb_classes = args.nb_classes
    mom = args.mom
    batch_size = args.batch_size
    lr = args.lr


#%% Import packages and configuration
import numpy as np
import tensorflow as tf
import h5py, os
import matplotlib.pyplot as plt
import imageio.v2 as imageio # for generating GIFs
import wandb
import subprocess
from tensorflow.keras.models import Model
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Activation, Flatten, Input, MaxPooling2D, BatchNormalization
from tensorflow.keras.backend import binary_crossentropy
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.optimize import linear_sum_assignment
from keras.src import backend
from skimage.transform import resize
from PIL import Image
from numba import cuda
import gc

def reset_gpu():
    # Clear Keras and TensorFlow sessions
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # Reinitialize GPU
    gc.collect()
    cuda.select_device(0)
    cuda.close()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    tf.config.experimental.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Enable GPU again

    # Enable memory growth for GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except:
            pass
        
    # Set eager execution to true for model training
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

# Configuration
reset_gpu()
command = "wandb login _"
subprocess.run(command, shell=True)

# Set random seeds and deterministic behaviour for reproducibility
# Dioooos la seed 42 es bonissimaaaa
# SEED = 42
# set_random_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

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
    l2_norm = tf.nn.l2_normalize(out, axis=-1)
    sim_mat = tf.matmul(l2_norm, l2_norm, transpose_b=True)

    cluster_l1 = Model(inputs=[inp_img], outputs=[out])
    cluster_l2 = Model(inputs=[inp_img], outputs=[l2_norm])
    model = Model(inputs=[inp_img], outputs=[out, sim_mat])

    return cluster_l1,cluster_l2,model

#%% Defining the functions

# plot mnist samples in groups of 10
def plot_samples(data_samples,first):
    if first:
        for j in range(3):
            if j % 3 == 0:
                ig, axs = plt.subplots(1, 3)
            axs[j % 3].imshow(data_samples[j, :, :])
            axs[j %3].set_axis_off()
            if j % 3 == 2 or j == 129:
                plt.tight_layout()
                plt.show()

def plot_matrices(cells,label_feat_norm,pos_loc_mask,neg_loc_mask,y_true,epoch):
    fig, axs = plt.subplots(1,4,figsize = (10,5))
    axs[0].imshow(label_feat_norm.numpy()[:cells,:cells],cmap="coolwarm")
    axs[0].set_title('sim_mat_pred')
    axs[1].imshow(pos_loc_mask[:cells,:cells],cmap="coolwarm",vmax = 1.0)
    axs[1].set_title('y_true')
    axs[2].imshow(neg_loc_mask[:cells,:cells],cmap="gray")
    axs[2].set_title('pos_loc_mask')
    axs[3].imshow(y_true[:cells,:cells],cmap="gray")
    axs[3].set_title('neg_loc_mask')
    fig.suptitle('Iteration %d of epoch %d' %(iteration if iteration==1 else iteration-1,epoch), y = 0.85)
    [ax.set_axis_off() for ax in axs]
    plt.tight_layout()
    plt.show()

# function to preprocess the MNIST dataset
def preprocess_data(X_data,y_data,mode, normalize= False):
    index = np.arange(X_data.shape[0])
    np.random.shuffle(index)
    X_data = X_data[index]
    y_data = y_data[index]
    if mode == "training":
        for i in range(X_data.shape[3]):
            mean_image = np.mean(X_data[:,:,:,i])
            sd = np.std(X_data[:,:,:,i])
            if normalize and sd!= 0:
                X_data[:, :, :, i] -= mean_image
                X_data[:, :, :, i] /= sd

    X_data = X_data/np.max(X_data)

    return X_data,y_data

# Get balanced data batches
def get_data_batch(batch_size, data, labels):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    samples_per_class = max(1, batch_size // num_classes)

    balanced_batch_data = []
    balanced_batch_labels = []
    balanced_indices = []

    for label in unique_labels:
        class_indices = np.where(shuffled_labels == label)[0]
        np.random.shuffle(class_indices)
        selected_indices = class_indices[:samples_per_class]
        balanced_batch_data.extend(shuffled_data[selected_indices])
        balanced_batch_labels.extend(shuffled_labels[selected_indices])
        balanced_indices.extend(indices[selected_indices])

    balanced_batch_data = np.array(balanced_batch_data)
    balanced_batch_labels = np.array(balanced_batch_labels)
    balanced_indices = np.array(balanced_indices)

    final_shuffle_indices = np.arange(len(balanced_indices))
    np.random.shuffle(final_shuffle_indices)
    balanced_batch_data = balanced_batch_data[final_shuffle_indices]
    balanced_batch_labels = balanced_batch_labels[final_shuffle_indices]
    final_indices = balanced_indices[final_shuffle_indices]

    return balanced_batch_data, balanced_batch_labels, final_indices

# Hungarian algorithm for clustering accuracy
def clustering_acc(y_true, y_pred):
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Calculate the indices using linear_sum_assignment after updating the matrix w
    ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size

    return acc,ind

# Defining the loss function
@tf.function
def loss_function(y_true, y_pred):
    global iteration, u_thres, l_thres, epoch

    # Masks for thresholding the similarity matrix and ommiting samples un between
    pos_loc = tf.greater(y_true, u_thres, name='greater')
    neg_loc = tf.less(y_true, l_thres, name='less')

    # Converting to float32
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)

    # alternative loss
    true_loc = pos_loc_mask + neg_loc_mask
    nb_selected = tf.reduce_sum(true_loc)
    y_true = tf.where(y_true > u_thres, 1.0, 0.0)
    loss_sum =  tf.reduce_sum(true_loc * binary_crossentropy(y_true, y_pred), axis=1) / nb_selected

    # plotting
    if (iteration-1)%500 ==0:
        print(u_thres, l_thres)
        # plot_matrices(40,y_pred,y_true,pos_loc_mask,neg_loc_mask,epoch)

    return loss_sum
    
#%% Importing the dataset

# images_Fe_50 = []
# images_Fe_10 = []
# images_Fe_5 = []
# images_Fe_1 = []
# images_Sn_50 = []
# images_Sn_10 = []
# images_Sn_5 = []
# images_Sn_1 = []
# images_Fe_100 = []
# images_Sn_100 = []

# images_list = [images_control, images_Fe_1, images_Fe_5, images_Fe_10, images_Fe_50, images_Fe_100, images_Ti_1, images_Ti_5,
#                images_Ti_10,images_Ti_50, images_Ti_100, images_Sn_1, images_Sn_5, images_Sn_10, images_Sn_50, images_Sn_100]

# Images path
conditions_list = ["Control","1Ti","5Ti","10Ti","50Ti","100Ti"]
files_path =  "/content/drive/MyDrive/fotos_immunos_norm"

# List to store the loaded images
images_Ti_100 = []
images_Ti_50 = []
images_Ti_10 = []
images_Ti_5 = []
images_Ti_1 = []
images_control = []
images_list = [images_control, images_Ti_1, images_Ti_5, images_Ti_10, images_Ti_50, images_Ti_100]

# Directory containing TIFF images
for i, cond in enumerate(conditions_list):
        path = os.path.join(files_path, "ML_A549_"+cond+"_PhaD_20X_RGB")
        for filename in os.listdir(path):
            if filename.endswith(".tif"):
                file_path = os.path.join(path, filename)
                tiff_image = np.array(Image.open(file_path))
                image = resize(tiff_image, (256, 256,3))
                images_list[i].append(image)

#%% Initialize network, parameters and optimizer

# Creating train labels for materials and concentration

yC = np.zeros(len(images_control))
y1 = 1 * np.ones(len(images_Ti_1))
y5 = 2 * np.ones(len(images_Ti_5))
y10 = 3 * np.ones(len(images_Ti_10))
y50 = 4 * np.ones(len(images_Ti_50))
y100 = 5 * np.ones(len(images_Ti_100))
y_train_C = np.hstack((yC, y1, y5, y10, y50, y100)).astype(int)

# Creating train dataset for materials and concentration
X_train_C = np.vstack((images_control, images_Ti_1, images_Ti_5, images_Ti_10, images_Ti_50, images_Ti_100))

# defining X_train and number of clusters
X_train = X_train_C
y_train = y_train_C
X_train, y_train = preprocess_data(X_train,y_train,"training",normalize=False)

# define input shape and thresholds
img_rows, img_cols, img_channels = 256, 256, 3
inp_shape = (img_rows, img_cols, img_channels)

# Initializing image data generator
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.09, height_shift_range=0.09,
    channel_shift_range=0.05, horizontal_flip=True,rescale=0.975, zoom_range=[0.95,1.05])

# parameters
epoch = 1
u_thres = 0.99
l_thres = 0.75
total_epochs = 50
trainBN = True
eta = (u_thres-l_thres)/total_epochs/2
nb = 64

# initialize the network and optimizer
cluster_l1,cluster_l2,model = ConvNetwork(inp_shape, nb_classes, trainBN, mom)
model.compile(optimizer=RMSprop(lr), loss=["binary_crossentropy", loss_function], loss_weights=[1,10])

# cositas
print_nums = False

# Start a run, tracking hyperparameters
wandb.init(project="DAC_immunos_C", name = "{}nc_{}-{}lr_N".format(nb_classes,lr,0.1*lr),
           # track hyperparameters and run metadata with wandb.config
           config={"nb_clusters": nb_classes,"BN momentum": mom,"BN learnable": trainBN,"batch_size": batch_size,
                   "u_thres": u_thres,"l_thres": l_thres,"lr": lr,"loss1": "binary_crossentropy",
                   "loss2": "custom_loss","epochs": total_epochs})

# paths
path = os.path.join(r"/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/immunos_TFG/DAC_immunos_concentration", "%d_clusters" %nb_classes)

if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created successfully.")

# Training loop

# Initialize the lists for the outputs and the network
accs = []
outputs = []
labels = []

index = np.arange(X_train.shape[0])
index_loc = np.arange(nb)
iteration = 0

# Perform training loop under condition
for e in range(total_epochs):
    np.random.shuffle(index)
    # Adjust learning rate after each epoch
    if e==35:
      lr = lr*0.1
      model.compile(optimizer=RMSprop(lr), loss=["binary_crossentropy", loss_function], loss_weights=[1,10])

    for i in range(X_train.shape[0]//nb): #numero de 64 que hiha al dataset
        data_samples,data_labels,_ = get_data_batch(72, X_train, y_train)
        Y = cluster_l2(data_samples, training = True).numpy()
        Y_l1 = cluster_l1(data_samples, training = True).numpy()
        Ybatch = np.dot(Y,Y.T)

        plot_samples(data_samples, print_nums)
        # print_nums = False

        for j in range(10):
            np.random.shuffle(index_loc)

            for k in range(nb//batch_size): # quants de batch sizes hiha
                address = index_loc[np.arange(k*batch_size,(k+1)*batch_size)]
                X_batch = data_samples[address]
                Y_batch = Ybatch[address,:][:,address]
                Y_batch_l1 = Y_l1[address]

                for it, X_batch_i in enumerate(datagen.flow(X_batch, batch_size=batch_size,shuffle=False)):
                    loss = model.train_on_batch([X_batch_i],[Y_batch_l1, Y_batch])
                    if it==0:
                        break

                iteration += 1
        print('Epoch: %d, batch: %d/%d, loss: %f, loss1: %f, loss2: %f, nb1: %f'%
              (e+1,i+1,X_train.shape[0]//nb, loss[0],loss[1],loss[2],np.mean(Ybatch)))

    # Run testing every epoch
    iteration = 0

    pred_cluster = []
    pred_labels = []
    len_range = int(len(X_train)/batch_size)
    for i in range(len_range):
        Xbatch = X_train[i*batch_size:(i+1)*batch_size]
        pred_cluster_i = cluster_l1(Xbatch, training = True)
        pred_labels_i = np.argmax(pred_cluster_i, axis=1)
        pred_cluster.append(pred_cluster_i)
        pred_labels.append(pred_labels_i)
    pred_labels = np.array(np.concatenate(pred_labels))
    pred_cluster = np.array(np.concatenate(pred_cluster))

    pred_cluster2 = cluster_l1.predict(X_train, verbose = 0)
    pred_labels2 = np.argmax(pred_cluster2, axis=1)

    acc,_ = clustering_acc(y_train[:len(pred_labels)], pred_labels)
    acc2,_ = clustering_acc(y_train, pred_labels2)

    accs.append(acc2)
    outputs.append(pred_cluster)
    labels.append(y_train[:len(pred_labels)])
    wandb.log({"accuracy1": acc, "accuracy2": acc2, "loss1": loss[0], "loss2": loss[1], "loss3": loss[2]})
    print('ACC at epoch %d is %.2f%%.' % (epoch, 100 * acc))
    print('ACC2 at epoch %d is %.2f%%.' % (epoch, 100 * acc2))

    # Update parameters
    u_thres -= eta
    l_thres += eta
    epoch += 1

#%% Saving results

# Save the outputs
path_out_folder = os.path.join(path,"DAC_outputs")
if not os.path.exists(path_out_folder):
    os.makedirs(path_out_folder)
    
code_iteration = 1
for file in os.listdir(path_out_folder):
    if file.endswith('.h5'):
        code_iteration += 1
path_out_file = os.path.join(path_out_folder,"outputs_iteration{}.h5".format(code_iteration))

file = h5py.File(path_out_file,'w')
file.create_dataset('accuracies', data = accs)
file.create_dataset('outputs', data = outputs)
file.create_dataset('labels', data = labels)
file.close()

# Save model at last epoch
path_weights = os.path.join(path,"DAC_weights")
if not os.path.exists(path_weights):
    os.makedirs(path_weights)
model.save_weights(os.path.join(path_weights,"DAC_weights_{}clusters_iter{}.h5".format(nb_classes, code_iteration)))
print("Model saved at epoch {}".format(epoch))

#%% Visualizing training process GIF

# reading the outputs file
with h5py.File(path_out_file, 'r') as hf:
    acc = hf['/accuracies'][:]  # clustering accuracies in all epochs
    out = hf['/outputs'][:]  # predictions in all epochs
    labs = hf['/labels'][:] # true labels

outputs = out.copy()

colors = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0], [1, 0, 1],[0, 1, 1],
                   [0, 0, 0],[0, 0.5, 0.5],[0.5, 0, 0.5],[0.5, 0.5, 0]])

k = np.finfo(float).eps

for i in range(outputs.shape[0]):
    labels_one = labs[i,:]
    one = outputs[i,:,:].reshape(outputs.shape[1], outputs.shape[2])
    # Calculate the sum along the second dimension (axis=1)
    one_sum = np.sum(one, axis=1, keepdims=True)
    # Replicate the sum along the second dimension to match the shape of one
    one_sum_replicated = np.tile(one_sum, (1, one.shape[1]))
    # Perform element-wise division
    one_normalized = one / one_sum_replicated
    # nb_classes = one.shape[1]
    theta = np.arange(0, 360, 360/nb_classes) / 180 * np.pi #36 degrees between points to have 10 points (decagon)
    points = np.column_stack([np.sin(theta), np.cos(theta)])  # high dim vectors to 2D points
    onet = np.dot(one_normalized, points)

    plt.figure()
    for j in range(onet.shape[0]):
        plt.plot(onet[j, 0], onet[j, 1], '.', linewidth=1, markersize=6, color=colors[labels_one[j]])

    plt.axis([-1, 1, -1.1, 1.3])
    plt.gca().set_position([0, 0, 1, 1])
    plt.title("Learning proccess mapped to regular polygon")
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_facecolor('w')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Increase the top margin
    path_learning = os.path.join(path, "DAC_learning")
    if not os.path.exists(path_learning):
        os.makedirs(path_learning)
    plt.savefig(os.path.join(path_learning, "epoch%d.png" % i))
    plt.show()
    plt.close()

# Create a GIF of the learning process
image_dir = os.path.join(path, "DAC_learning")
images = sorted(os.listdir(image_dir))[1:] # List the image files
image_paths = [os.path.join(image_dir, img) for img in images] # Create a list to store image paths
gif_path = os.path.join(path,"learning_gif.gif")
frame_rate = 2.5

with imageio.get_writer(gif_path, mode='I', fps=frame_rate) as writer:
    for image_path in image_paths:
        image = imageio.imread(image_path)
        writer.append_data(image)

print(f"GIF created successfully at: {gif_path}")

#%% Visualize learning curve

epochs = range(1, len(accs) + 1)

plt.plot(epochs, accs, 'b', label='Training accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.yticks([i/10 for i in range(5, 11)])
plt.grid(True)
plt.savefig(os.path.join(path,"learning_curve_iter{}.png".format(code_iteration)))
plt.show()

# finish the run
wandb.finish()