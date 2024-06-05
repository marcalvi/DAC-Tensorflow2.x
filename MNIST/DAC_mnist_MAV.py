#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Mon Mar 18 13:42:29 2024
@author: marcalbesa

"""

#%% Imoport packages and configuration
import numpy as np
import tensorflow as tf
import random, h5py, os, datetime
import matplotlib.pyplot as plt
import imageio.v2 as imageio # for generating GIFs
from tensorflow.keras.models import Model
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Activation, Flatten, Input, MaxPooling2D, BatchNormalization 
from tensorflow.keras.initializers import he_normal, identity
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
from scipy.optimize import linear_sum_assignment

# Set random seeds and deterministic behaviour for reproducibility
# Dioooos la seed 42 es bonissimaaaa
# SEED = 42
# set_random_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

# # Set eager execution to true for model training
tf.config.run_functions_eagerly(True)

# Verify GPU environment is active
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("GPU Name:", gpu.name)
else:
    print("No GPU devices found.")

#%% Creating the ConvNet

def ConvNetwork(input_shape, nb_classes):    
    # First convolutional block
    inp_img = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(inp_img)
    bn1 = BatchNormalization(axis=-1, epsilon=1e-5)(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(relu1)
    bn2 = BatchNormalization(axis=-1, epsilon=1e-5)(conv2)
    relu2 = Activation('relu')(bn2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(relu2)
    bn3 = BatchNormalization(axis=-1, epsilon=1e-5)(conv3)
    relu3 = Activation('relu')(bn3)
    
    # First max pool block
    max1 = MaxPooling2D((2, 2), (2, 2),padding='same')(relu3)
    bn4 = BatchNormalization(axis=-1, epsilon=1e-5)(max1)
    
    # Second convolutional block
    conv4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(bn4)
    bn5 = BatchNormalization(axis=-1, epsilon=1e-5)(conv4)
    relu4 = Activation('relu')(bn5)
    conv5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(relu4)
    bn6 = BatchNormalization(axis=-1, epsilon=1e-5)(conv5)
    relu5 = Activation('relu')(bn6)
    conv6 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(relu5)
    bn7 = BatchNormalization(axis=-1, epsilon=1e-5)(conv6)
    relu6 = Activation('relu')(bn7)
    
    # Second max pool block
    max2 = MaxPooling2D((2, 2), (2, 2), padding='same')(relu6) 
    bn8 = BatchNormalization(axis=-1, epsilon=1e-5)(max2) 
    
    # Classifier block
    conv7 = Conv2D(10, (1, 1), strides=(1, 1), kernel_initializer= he_normal(), padding="valid")(bn8)
    bn9 = BatchNormalization(axis=-1, epsilon=1e-5)(conv7)
    relu7 = Activation('relu')(bn9)
    
    # Average Pooling
    avg = AveragePooling2D((2, 2), (2, 2), padding='same')(relu7)
    bn10 = BatchNormalization(axis=-1, epsilon=1e-5)(avg)
    
    # First dense block
    flat = Flatten()(bn10)
    dense1 = Dense(nb_classes, kernel_initializer= identity())(flat)
    bn11 = BatchNormalization(axis=-1, epsilon=1e-5)(dense1)
    relu8 = Activation('relu')(bn11)
    
    # Second dense block
    dense2 = Dense(nb_classes, kernel_initializer= identity())(relu8)
    bn12 = BatchNormalization(axis=-1, epsilon=1e-5)(dense2) 
    relu9 = Activation('relu')(bn12)
    out = Activation('softmax')(relu9)
    
    model = Model(inputs=inp_img, outputs=out)

    return model

#%% Defining the functions

# plot mnist samples in groups of 10
def plot_samples(data_samples,first):
    if first:
        for j in range(10):
            if j % 10 == 0:
                ig, axs = plt.subplots(1, 10)
            axs[j % 10].imshow(data_samples[j, :, :, 0], cmap="gray")
            axs[j % 10].set_axis_off()
            if j % 10 == 9 or j == 129:
                plt.tight_layout()
                plt.show()

# function to preprocess the MNIST dataset
def preprocess_mnist(X_data,mode):
    global mean_image_train
    X_expanded = np.expand_dims(X_data, axis=-1)
    X_norm = X_expanded.astype('float32') / 255.
    if mode == "training":
        mean_image_train = np.mean(X_norm[:, :, :, 0])
        #guardar a un fitxer de text
    X_norm[:, :, :, 0] -= mean_image_train
    X_norm[X_norm < 0] = 0
    
    return X_norm

def get_mnist_batch(batch_size, mnist_data, mnist_labels):
    # Shuffle the samples and store the order
    indices = np.arange(mnist_data.shape[0])
    np.random.shuffle(indices)
    batch = mnist_data[indices]
    y_batch = mnist_labels[indices]

    # Select a batch of data
    batch_data = batch[:batch_size]
    batch_label = y_batch[:batch_size]
    return batch_data, batch_label

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

# # Defining the loss function
sim_mats = []
@tf.function
def loss_function(y_pred, u_thres, l_thres):
    global iteration
    eps = 1e-10  # term added for numerical stability of log computations
    
    # get similarity matrix
    label_feat_norm = tf.nn.l2_normalize(y_pred, axis=1)
    sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)
    
    # Masks for thresholding the similarity matrix and ommiting samples un between
    pos_loc = tf.greater(sim_mat, u_thres, name='greater')
    neg_loc = tf.less(sim_mat, l_thres, name='less')
    
    # Converting to float32
    pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
    neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)
    
    # Applying logarithm for computing the entropy and thresholding 
    pos_entropy = tf.multiply(-tf.math.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
    neg_entropy = tf.multiply(-tf.math.log(tf.clip_by_value(1 - sim_mat, eps, 1.0)), neg_loc_mask)
    
    # Defining loss as the sum of positive and negative entropy
    loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)
    
    # plotting
    if (iteration-1)%500 ==0:
        sim_mats.append(sim_mat)
        cells = 10
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(label_feat_norm.numpy()[:cells,:],cmap="coolwarm")
        axs[0].set_title('Label features')
        axs[1].imshow(sim_mat.numpy()[:cells,:cells],cmap="coolwarm")
        axs[1].set_title('Similarity matrix')
        axs[2].imshow(pos_loc_mask.numpy()[:cells,:cells],cmap="gray",vmax=pos_loc_mask.numpy().max())
        axs[2].set_title('Positive centers')
        axs[3].imshow(neg_loc_mask.numpy()[:cells,:cells],cmap="gray",vmax=neg_loc_mask.numpy().max())
        axs[3].set_title('Negative centers')
        fig.suptitle('Iteration %d of epoch %d' %(iteration if iteration==1 else iteration-1,epoch), y = 0.77)
        [ax.set_axis_off() for ax in axs]
        plt.tight_layout()
        plt.show()
        
    return loss_sum

#%% Initialize network, parameters and optimizer

# defining number of clusters
nb_classes = 10

# define input shape
img_rows, img_cols, img_channels = 28, 28, 1
inp_shape = (img_rows, img_cols,img_channels)

# import mnist
(X_train, y_train), (_,_) = mnist.load_data()
X_train = preprocess_mnist(X_train,"training")

# parameters
batch_size = 256
lamda = 0
epoch = 1
u_thres = 0.95
l_thres = 0.455
lr = 0.001

# initialize the network and optimizer
ConvNet = ConvNetwork(inp_shape, nb_classes)
optimizer = RMSprop(lr)

# cositas
print_nums = True

# paths
path = os.path.join(r"/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/mnist/", "%d_clusters" %nb_classes)

if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created successfully.")

#%% Training loop

# Initialize the lists for the outputs
accs = []
outputs = []
labels = []

# Perform training loop under condition
while u_thres > l_thres:
    for iteration in range(1, 1001): # 1000 iterations is roughly 1 epoch
        data_samples,batch_label = get_mnist_batch(batch_size, X_train, y_train)
        plot_samples(data_samples, print_nums)
        print_nums = False

        # Training step
        with tf.GradientTape() as tape:
            predictions = ConvNet(data_samples,training = True)
            loss_value = loss_function(predictions, u_thres, l_thres)
          
        # Compute gradients and update weights
        gradients = tape.gradient(loss_value, ConvNet.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ConvNet.trainable_variables))
        
        if iteration % 50 == 0:
            tf.print("training loss at iteration ",str(iteration),"is :", 100*loss_value)
            
    # Run testing every epoch
    data_samples, data_labels = get_mnist_batch(2048, X_train, y_train)
    pred_cluster = ConvNet.predict(data_samples)
    pred_labels = np.argmax(pred_cluster, axis=1)
    acc,_ = clustering_acc(data_labels, pred_labels)
    accs.append(acc)
    outputs.append(pred_cluster)
    labels.append(data_labels)
    print('ACC at epoch %d is %.2f%%.' % (epoch, 100 * acc))

    # Update parameters
    lamda += 1.1 * 0.009
    u_thres -= lamda
    l_thres += 0.1 * lamda
    epoch += 1

#%% Saving results

# Save model at last epoch
path_weights = os.path.join(path,"DAC_weights")
if not os.path.exists(path_weights):
    os.makedirs(path_weights)
ConvNet.save_weights(os.path.join(path_weights,"DAC_weights_{}clusters.h5".format(nb_classes)))
print("Model saved at epoch 10")

# Save the outputs
path_out_folder = os.path.join(path,"DAC_outputs")
if not os.path.exists(path_out_folder):
    os.makedirs(path_out_folder)
code_iteration = 1
for file in os.listdir(path_out_folder):
    if file.endswith('.h5'):
        code_iteration += 1
path_out_file = os.path.join(path_out_folder,"outputs_mnist_iteration{}.h5".format(code_iteration))
    
file = h5py.File(path_out_file,'w')
file.create_dataset('accuracies', data = accs)
file.create_dataset('outputs', data = outputs)
file.create_dataset('labels', data = labels)
file.close()

#%% Test mode

# Reload MNIST dataset and preprocess it
(_,_), (X_test, y_test) = mnist.load_data()
X_test = preprocess_mnist(X_test,"test")

# Reload the model and load weights
ConvNet = ConvNetwork(inp_shape, nb_classes)
ConvNet.load_weights(os.path.join(path_weights,"DAC_weights_{}clusters.h5".format(nb_classes)))

# Find images with label 3
indices_3 = np.where(y_test == 1)[0]
imatges = X_test[indices_3]
data = imatges[:20,:,:,:]

# imatge_expand = np.expand_dims(imatge, axis=0)  # Add dimension on the left
pred_cluster = ConvNet(data,training = False)
pred_label = np.argmax(pred_cluster, axis=1)

print("Predicted labels: " + str(pred_label))

#%% Mapping results

values_map = {}

for i in range(nb_classes):
    indices_3 = np.where(y_test == i)[0]
    imatges = X_test[indices_3]
    data = imatges[:50,:,:,:]
    pred_cluster = ConvNet.predict(data) 
    pred_label = np.argmax(pred_cluster, axis=1)[0]
    print(i)
    values_map["Cluster %d"%i] = pred_label
    
print(values_map)

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
nb_points = 1024  # number of points

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
    for j in range(nb_points):
        plt.plot(onet[j, 0], onet[j, 1], '.', linewidth=1, markersize=6, color=colors[labels_one[j]])
        
    plt.axis([-1, 1, -1.1, 1.3])
    plt.gca().set_position([0, 0, 1, 1])
    plt.title("Learning proccess mapped to regular decagon")
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
plt.savefig(os.path.join(path,"learning_curve.png"))
plt.show()

#%% play sound
import subprocess

def speak(text):
    subprocess.call(['say', text])

if __name__ == "__main__":
    text_to_speak = "ENTRENAMIENTO COMPLETADO"
    speak(text_to_speak)

