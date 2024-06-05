#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data: Tue Apr  2 12:51:52 2024
@author: marcalbesa

"""

#%% Import packages and configuration
import numpy as np
import tensorflow as tf
import h5py, os
import matplotlib.pyplot as plt
import imageio.v2 as imageio # for generating GIFs
from tensorflow.keras.models import Model
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Activation, Flatten, Input, MaxPooling2D, BatchNormalization 
from tensorflow.keras.backend import binary_crossentropy
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.optimize import linear_sum_assignment
from keras.src import backend

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
        tf.config.experimental.set_memory_growth(gpu, True)

else:
    print("No GPU devices found.")
    
class MyBatchNorm(tf.keras.layers.BatchNormalization):

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

def ConvNetwork2(input_shape, nb_classes):    
    # First convolutional block
    inp_img = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(inp_img)
    bn1 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu1)
    bn2 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv2)
    relu2 = Activation('relu')(bn2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu2)
    bn3 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv3)
    relu3 = Activation('relu')(bn3)
    
    # First max pool block
    max1 = MaxPooling2D((2, 2), (2, 2),padding='same')(relu3)
    
    # Second convolutional block
    conv4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max1)
    bn5 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv4)
    relu4 = Activation('relu')(bn5)
    conv5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu4)
    bn6 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv5)
    relu5 = Activation('relu')(bn6)
    conv6 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu5)
    bn7 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv6)
    relu6 = Activation('relu')(bn7)
    
    # Second max pool block
    max2 = MaxPooling2D((2, 2), (2, 2), padding='same')(relu6) 
    
    # Classifier block
    conv7 = Conv2D(10, (1, 1), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max2)
    bn9 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(conv7)
    relu7 = Activation('relu')(bn9)
    
    # Average Pooling
    avg = AveragePooling2D((2, 2), (2, 2), padding='same')(relu7)
    
    # First dense block
    flat = Flatten()(avg)
    dense1 = Dense(nb_classes, kernel_initializer= 'identity')(flat)
    bn11 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(dense1)
    relu8 = Activation('relu')(bn11)
    
    # Second dense block
    dense2 = Dense(nb_classes, kernel_initializer= 'identity')(relu8)
    bn12 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = False)(dense2) 
    relu8 = Activation('relu')(bn12)
    out = Activation('softmax')(relu8)
    l2_norm = tf.nn.l2_normalize(out, axis=-1)
    sim_mat = tf.matmul(l2_norm, l2_norm, transpose_b=True)

    cluster_l1 = Model(inputs=[inp_img], outputs=[out])
    cluster_l2 = Model(inputs=[inp_img], outputs=[l2_norm])
    model = Model(inputs=[inp_img], outputs=[out, sim_mat])

    return cluster_l1,cluster_l2,model

def ConvNetwork(input_shape, nb_classes):    
    # First convolutional block
    inp_img = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(inp_img)
    bn1 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv1)
    relu1 = Activation('relu')(bn1)
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu1)
    bn2 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv2)
    relu2 = Activation('relu')(bn2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu2)
    bn3 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv3)
    relu3 = Activation('relu')(bn3)
    
    # First max pool block
    max1 = MaxPooling2D((2, 2), (2, 2),padding='same')(relu3)
    
    # Second convolutional block
    conv4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max1)
    bn5 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv4)
    relu4 = Activation('relu')(bn5)
    conv5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu4)
    bn6 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv5)
    relu5 = Activation('relu')(bn6)
    conv6 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(relu5)
    bn7 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv6)
    relu6 = Activation('relu')(bn7)
    
    # Second max pool block
    max2 = MaxPooling2D((2, 2), (2, 2), padding='same')(relu6) 
    
    # Classifier block
    conv7 = Conv2D(10, (1, 1), strides=(1, 1), kernel_initializer= 'he_normal', padding="valid")(max2)
    bn9 = MyBatchNorm(epsilon=1e-5, trainable = True)(conv7)
    relu7 = Activation('relu')(bn9)
    
    # Average Pooling
    avg = AveragePooling2D((2, 2), (2, 2), padding='same')(relu7)
    
    # First dense block
    flat = Flatten()(avg)
    dense1 = Dense(nb_classes, kernel_initializer= 'identity')(flat)
    bn11 = MyBatchNorm(epsilon=1e-5, trainable = True)(dense1)
    relu8 = Activation('relu')(bn11)
    
    # Second dense block
    dense2 = Dense(nb_classes, kernel_initializer= 'identity')(relu8)
    bn12 = MyBatchNorm(momentum = 0, epsilon=1e-5, trainable = True)(dense2) 
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
        for j in range(10):
            if j % 10 == 0:
                ig, axs = plt.subplots(1, 10)
            axs[j % 10].imshow(data_samples[j, :, :],vmax=1)
            axs[j % 10].set_axis_off()
            if j % 10 == 9 or j == 129:
                plt.tight_layout()
                plt.show()

def plot_matrices(cells,label_feat_norm,pos_loc_mask,neg_loc_mask,y_true):
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
def preprocess_cifar10(X_data,y_data,mode, normalize= False):    
    index = np.arange(X_data.shape[0])
    np.random.shuffle(index)
    X_data = X_data[index]
    y_data = y_data[index]
    if mode == "training":
        for i in range(X_data.shape[3]):
            mean_image = np.mean(X_data[:,:,:,i])
            sd = np.std(X_data[:,:,:,i])
    if normalize:
        X_data[:, :, :, i] -= mean_image
        X_data[:, :, :, i] /= sd
    
    X_data = X_data/np.max(X_data)
    
    return X_data,y_data

def get_cifar10_batch(batch_size, X, Y):
    # Shuffle the samples and store the order
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    batch = X[indices]
    y_batch = Y[indices]

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

# Defining the loss function
@tf.function
def loss_function(y_true, y_pred):
    global iteration, u_thres, l_thres
        
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
        plot_matrices(40,y_pred,y_true,pos_loc_mask,neg_loc_mask)
        
    return loss_sum

#%% Initialize network, parameters and optimizer

# defining number of clusters
nb_classes = 10

# define input shape
img_rows, img_cols, img_channels = 32, 32, 3
inp_shape = (img_rows, img_cols, img_channels) 

# import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = np.vstack((X_train, X_test))
y_train = np.vstack((y_train, y_test)).astype('int64')
X_train, y_train = preprocess_cifar10(X_train,y_train,"training",normalize=False)

# Initializing image data generator 
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.09, height_shift_range=0.09, 
    channel_shift_range=0.05, horizontal_flip=True,rescale=0.975, zoom_range=[0.95,1.05])

# parameters
batch_size = 32
epoch = 1
u_thres = 0.99
l_thres = 0.75
total_epochs = 50
eta = (u_thres-l_thres)/(2*total_epochs)
lr = 0.001

# initialize the network and optimizer
rmsprop = RMSprop(lr)
cluster_l1,cluster_l2,model = ConvNetwork(inp_shape, nb_classes)
model.compile(optimizer=RMSprop(lr), loss=["binary_crossentropy", loss_function], loss_weights=[1,10])

# cositas
print_nums = True

# paths
path = os.path.join(r"/Users/marcalbesa/Desktop/TFG/xarxa/DAC-MarcAlbesa/cifar10/", "%d_clusters" %nb_classes)

if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created successfully.")
    
    
#%% Training loop 

# Initialize the lists for the outputs and the network
accs = []
outputs = []
labels = []

index = np.arange(X_train.shape[0])
index_loc = np.arange(1000)
iteration = 0


# Perform training loop under condition
for e in range(31,total_epochs+1):
    np.random.shuffle(index)
    
    for i in range(X_train.shape[0]//1000): #numero de 1000s que hiha al dataset
        data_samples = X_train[index[i*1000:(i+1)*1000]]
        Y = cluster_l2(data_samples, training = True).numpy()
        Y_l1 = cluster_l1(data_samples, training = True).numpy()
        Ybatch = np.dot(Y,Y.T)
        
        plot_samples(data_samples, print_nums)
        print_nums = False
        
        for j in range(10):
            np.random.shuffle(index_loc)
            
            for k in range(data_samples.shape[0]//batch_size): #quants de batch sizes hiha
                address = index_loc[np.arange(k*batch_size,(k+1)*batch_size)]
                X_batch = data_samples[address]
                Y_batch = Ybatch[address,:][:,address]
                Y_batch_l1 = Y_l1[address]
                
                for it, X_batch_i in enumerate(datagen.flow(X_batch, batch_size=batch_size,shuffle=False)):
                    loss = model.train_on_batch([X_batch_i],[Y_batch_l1, Y_batch])
                    if it==0:
                        break
                    
                iteration += 1
        if i%10==0:
            print('Epoch: %d, batch: %d/%d, loss: %f, loss1: %f, loss2: %f, nb1: %f'%
                  (e+1,i+1,X_train.shape[0]//1000, loss[0],loss[1],loss[2],np.mean(Ybatch)))  
    # Run testing every epoch
    iteration = 0
    data_samples, data_labels = get_cifar10_batch(2048, X_train, y_train)
    pred_cluster_traintrue = cluster_l1(data_samples, training = True)
    pred_cluster_predict = cluster_l1(data_samples, training = False)
    pred_labels1 = np.argmax(pred_cluster_traintrue, axis=1)
    pred_labels2 = np.argmax(pred_cluster_predict, axis=1)
    acc1,_ = clustering_acc(data_labels, pred_labels1)
    acc2,_ = clustering_acc(data_labels, pred_labels2)
    accs.append((acc1,acc2))
    outputs.append(pred_cluster_traintrue)
    labels.append(data_labels)
    print('ACC1 at epoch %d is %.2f%%.' % (epoch, 100 * acc1))
    print('ACC2 at epoch %d is %.2f%%.' % (epoch, 100 * acc2))
        
    # Releasing memory
    if epoch%2 ==0:
        # Save model at last epoch
        path_weights = os.path.join(path,"DAC_weights")
        if not os.path.exists(path_weights):
            os.makedirs(path_weights)
        model.save_weights(os.path.join(path_weights,"DAC_weights_{}clusters_epoch{}.h5".format(nb_classes,e)))
        print("Model saved at epoch {}".format(epoch))
        # Clear the TensorFlow session
        tf.keras.backend.clear_session()
        # rebuild the model
        cluster_l1,cluster_l2,model = ConvNetwork(inp_shape, nb_classes)
        model.compile(optimizer=RMSprop(lr), loss=["binary_crossentropy", loss_function], loss_weights=[1,10])
        model.load_weights(os.path.join(path_weights,"DAC_weights_{}clusters_epoch{}.h5".format(nb_classes,e)))
        
    # Update parameters
    print_nums = True
    u_thres -= eta
    l_thres += eta 
    epoch += 1

#%% Saving results

# Save model at last epoch
path_weights = os.path.join(path,"DAC_weights")
if not os.path.exists(path_weights):
    os.makedirs(path_weights)
model.save_weights(os.path.join(path_weights,"DAC_weights_{}clusters.h5".format(nb_classes)))
print("Model saved at epoch {}".format(epoch-1))

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

# Reload CIFAR10 dataset and preprocess it
(_, _), (X_test, y_test) = cifar10.load_data()
# X_test = preprocess_cifar10(x_test,"test")

# Reload the model and load weights
cluster_l1,cluster_l2,model = ConvNetwork(inp_shape, nb_classes)
model.load_weights(os.path.join(path,"DAC_weights/DAC_weights_{}clusters.h5".format(nb_classes)))

# Find images with label 3
indices_3 = np.where(y_test == 1)[0]
imatges = X_test[indices_3]
data = imatges[:20,:,:,:]

# imatge_expand = np.expand_dims(imatge, axis=0)  # Add dimension on the left
pred_cluster = cluster_l1(data,training = False)
pred_label = np.argmax(pred_cluster, axis=1)

print("Predicted labels: " + str(pred_label))


#%% Mapping results

values_map = {}

for i in range(nb_classes):
    indices_3 = np.where(y_test == i)[0]
    imatges = X_test[indices_3]
    data = imatges[:50,:,:,:]
    pred_cluster = cluster_l1.predict(data) 
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
accs = accs[:19]
epochs = range(1,len(accs) + 1)

plt.plot(epochs, [acc[0] for acc in accs], 'b', label='Training accuracy')
plt.plot(epochs, [acc[1] for acc in accs], 'r', label='Test accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(path,"learning_curve.png"))
plt.show()

#%% play sound
import subprocess

def speak(text):
    subprocess.call(['say', text])

if __name__ == "__main__":
    text_to_speak = "ANTRENAMENT COMPLETAT"
    speak(text_to_speak)
