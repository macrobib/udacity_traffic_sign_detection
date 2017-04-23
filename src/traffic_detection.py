import cv2
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
import random
from math import ceil
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import *
import sys
training_file = '../data/train.p'
testing_file = '../data/test.p'
validation_file = '../data/valid.p'

# Define the variables
model = None
x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_labels = tf.placeholder(tf.float32, [None, 43])
prob = tf.placeholder(tf.float32)
is_training = True

# Define the weights - xavier normal.
weights_xav = {
        'w1': tf.get_variable(shape=(5, 5, 3, 6), initializer=tf.contrib.layers.xavier_initializer(), name='W1'),
        'w2': tf.get_variable(shape=(5, 5, 6, 16),initializer=tf.contrib.layers.xavier_initializer(), name='W2'),
        'w3': tf.get_variable(shape=(8*8*16, 400), initializer=tf.contrib.layers.xavier_initializer(), name='W3'),
        'w4': tf.get_variable(shape=(400, 120), initializer=tf.contrib.layers.xavier_initializer(), name='W4'),
        'w5': tf.get_variable(shape=(120, 84), initializer=tf.contrib.layers.xavier_initializer(), name='W5'),
        'w6': tf.get_variable(shape=(84, 43),  initializer=tf.contrib.layers.xavier_initializer(), name='W6')
        }

# Define the weights - truncated initialization.
weights_trunc = {
        'w1': tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=0., stddev=0.1), name='W1'),
        'w2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=0., stddev=0.1), name='W2'),
        'w3': tf.Variable(tf.truncated_normal(shape=(8*8*16, 400), mean=0., stddev=0.1), name='W3'),
        'w4': tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0., stddev=0.1), name='W4'),
        'w5': tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0., stddev=0.1), name='W5'),
        'w6': tf.Variable(tf.truncated_normal(shape=(84, 43), mean=0., stddev=0.1), name='W6')
        }
# Selected weights
weights = weights_xav

#Define the biases.
biases = {
        'b1': tf.Variable(tf.truncated_normal([6], mean=0., stddev=0.1),name='B1'),
        'b2': tf.Variable(tf.truncated_normal([16], mean=0., stddev=0.1),name='B2'),
        'b3': tf.Variable(tf.truncated_normal([400], mean=0., stddev=0.1),name='B3'),
        'b4': tf.Variable(tf.truncated_normal([120], mean=0., stddev=0.1),name='B4'),
        'b5': tf.Variable(tf.truncated_normal([43], mean=0., stddev=0.1),name='B5')
        }

biases_s = {
        'b1': tf.Variable(tf.constant(0.1, shape=[6])),
        'b2': tf.Variable(tf.constant(0.1, shape=[16])),
        'b3': tf.Variable(tf.constant(0.1, shape=[400])),
        'b4': tf.Variable(tf.constant(0.1, shape=[120])),
        'b5': tf.Variable(tf.constant(0.1, shape=[84])),
        'b6': tf.Variable(tf.constant(0.1, shape=[43]))
        }

#Load the datasets.
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(validation_file, mode='rb') as f:
    validate = pickle.load(f)

# Save the images.
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
X_Validate, y_validate = validate['features'], validate['labels']

X_train.astype(np.float32)
X_Validate.astype(np.float32)
X_test.astype(np.float32)

print("Total length", len(y_train))
print("Labels are: ", y_train)
print("Test sample shape is: ", X_test.shape)
# Create a list of randomization function, which will be again called for a random
# count and random order.
data_augment_functions = {}
content = {}
# Open the label data and save it to an array.
with open('../data/signnames.csv', 'r') as file:
    csvfile = csv.reader(file)
    for line, row in enumerate(csvfile):
        if line != 0:
            content[int(row[0])] = [row[1], 0]

backup_content = content

for i in y_train:
    content[i][1] += 1

# content = content[1:]
# content_label = [x.split(',')[1].strip() for x in content]
# print(content_label)
# content_dict = {content[x].split(',')[1].strip():0 for x in range(1,len(content))}
# del(content[content_label[0]])

# print("Modified image size: ", dst.shape)
# print("Label of the data: {}".format(content[y_train[index]][0]))
# plt.figure(figsize=(1, 1))
# plt.imshow(dst)
# plt.show()
# Print a sample image.
# print("Sample data..", X_train)
print("Sample data shape..", X_train[0].shape)
def display_image(image):
    """Utility function to check images"""
    plt.figure(figsize=(1,1))
    plt.imshow(image)
    plt.show()

def random_image():
    global X_train
    global y_train
    val = random.randint(0, 400)
    print("Image label is: ", y_train[val])
    display_image(X_train[val])


def shuffle_data():
    "Shuffle data: Both training and testing"
    global X_train
    global y_train
    global X_Validate
    global y_validate
    global X_test
    global y_test

    row_train, _, _, _ = X_train.shape
    shuffle_index = np.arange(row_train)
    np.random.shuffle(shuffle_index)
    X_train = X_train[shuffle_index, :, :, :]
    y_train = y_train[shuffle_index, :]

    row_validate, _, _, _ = X_Validate.shape
    shuffle_index = np.arange(row_validate)
    np.random.shuffle(shuffle_index)
    X_Validate = X_Validate[shuffle_index, :, :, :]
    y_validate = y_validate[shuffle_index, :]

    row_test, _, _, _ = X_test.shape
    shuffle_index = np.arange(row_test)
    np.random.shuffle(shuffle_index)
    X_test = X_test[shuffle_index, :, :, :]
    y_test = y_test[shuffle_index, :]
    print("Shuffled test data shape: ", X_test.shape)
    print("Shuffle: ", row_train, row_test)

def one_hot_encode():
    """One hot encode the label data and return."""
    global y_train
    global y_validate
    global y_test
    binarize = preprocessing.LabelBinarizer()
    binarize.fit(y_train)
    y_train = binarize.transform(y_train)
    y_validate = binarize.transform(y_validate)
    y_test = binarize.transform(y_test)

    # Convert to float values.
    y_train    = y_train.astype(np.float32)
    y_validate = y_validate.astype(np.float32)
    y_test     = y_test.astype(np.float32)
    print("Hot encoded value: ", y_train[0])

def rotation(image, val=10):
    dst = np.array(image)
    row, col, _ = dst.shape
    rotate_by = random.randint(5, 5+val)
    M = cv2.getRotationMatrix2D((col / 2, row / 2), rotate_by, 1)
    dst = cv2.warpAffine(dst, M, (col, row))
    return dst


def translation(image, trans_range=3):
    row, col, _ = image.shape
    tr_x = trans_range*np.random.uniform() - trans_range/2
    tr_y = trans_range*np.random.uniform() - trans_range/2
    M = np.float32([[1,0,tr_x], [0, 1, tr_y]])
    dst = cv2.warpAffine(image, M, (col, row))
    return dst

def shear(image, shear_val=5):
    row, col, _ = image.shape
    point1 = np.float32([[5, 5], [20, 5], [5, 20]])
    p1 = 5 + shear_val*np.random.uniform() - shear_val/2
    p2 = 20 + shear_val*np.random.uniform() - shear_val/2
    point2 = np.float32([[p1, 5], [p2, p1], [5, p2]])

    M = cv2.getAffineTransform(point1, point2)
    dst = cv2.warpAffine(image, M, (col, row))
    return dst


def random_shift_color_channels(image, val=10, enable=True):
    dst = image
    enable = False
    if enable == True:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        dst = np.dstack((np.roll(R, random.randint(-2, val), axis=0),
                         np.roll(G, random.randint(-2, val), axis=1),
                         np.roll(B, random.randint(-2, val), axis=0)))
    return dst


def flip(image):
    """Randomly flip the image."""
    dst = image
    enable = False
    if enable == True:
        dst = cv2.flip(image, random.randint(0,1))
    return dst


def define_aug_functions():
    """Define a set of augmentation lambda functions"""
    global data_augment_functions
    # Random ranged rotate
    data_augment_functions[0] = rotation
    # shift image
    data_augment_functions[1] = translation
    # Random shear in range.
    data_augment_functions[2] = shear
    # Flip randomly
    data_augment_functions[3] = flip
    # Random shift color channels.
    data_augment_functions[4] = random_shift_color_channels


def augment_image(image):
    """Augment the data"""
    dst = image
    define_aug_functions()
    val = [i for i in range(3)]
    random.shuffle(val)
    for i in val:
        dst = data_augment_functions[i](image)
    return dst


def augment_data():
    global X_train
    global y_train
    global content
    data_index = 0
    print("Copying data..")
    list_X_train = X_train.tolist()
    list_y_train = y_train.tolist()
    labelwise_data = {i:5000 - content[i][1] for i in content.keys()}
    data_len = len(X_train)
    augment_count = sum(labelwise_data.values())
    # Seperate the data to individual labels.
    print("Start augmenting the data..")
    while augment_count > 0:
        for i in tqdm(range(data_len)):
            index = y_train[i]
            if labelwise_data[index] > 0:
                augment_count -= 1
                labelwise_data[index] -= 1
                image = augment_image(X_train[i])
                list_X_train.append(list(image))
                list_y_train.append(index)
        # repopulate content data structure.
    print("Creating a new set of numpy array.")
    X_train = np.array(list_X_train)
    y_train = np.array(list_y_train)
    print('shape of X-Train: {0} and y-train: {1}'.format(
        X_train.shape, y_train.shape))


    for i in y_train:
        backup_content[i][1] += 1
    print(backup_content)
    print("Finished: Augmenting data..")



def split_data():
    # Split the training data into training and validation sets.
    # D0 a 70-30 split of the training and validation data.
    global X_train
    global y_train
    global X_Validate
    global y_validate
    print("Start: Splitting data.")

    split_val = ceil(len(X_train) * 0.8)
    X_Validate = X_train[split_val: len(X_train)]
    y_validate = y_train[split_val: len(y_train)]
    X_train = X_train[0:split_val]
    y_train = y_train[0:split_val]
    print("Length of training set: ", np.array(y_train).shape)
    print("Length of validation set: ", np.array(y_validate).shape)

    # Check the count of the data present.
    print("Finish: Splitting data. ")
    for x in range(len(y_train)):
        content[y_train[x]][1] += 1
    print(content)


def visualize_data():
    """"Visualize the given data."""
    figure, ax = plt.subplots(figsize=(20, 40))
    ylen = np.arange(len(content))
    list_names = [list(content.values())[i][1] for i in range(len(content))]
    ax.barh(ylen, list_names, align='center', color='green')
    ax.set_yticklabels(content.keys())
    ax.set_yticks(ylen)
    ax.set_xlabel('No of Images')
    ax.set_title('Image distribution')
    figure.subplots_adjust(left=0.12)
    plt.show()

def standardization(image):
    """Standardize the given image."""
    mean = image.mean()
    stdev = image.std()
    return (image - mean)/stdev

def normalize(image):
    """Min-Max normalize the given image."""
    return (image - 128.0)/256.0

def normalize_data():
    """Normalize give data using histogram equalization."""
    global X_train
    global X_Validate
    global X_test

    for i in range(len(X_train)):
        image = X_train[i].astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        V_channel = image[:, :, 2]
        # Standardize the image.
        image[:, :, 2] = standardization(V_channel)
        # Normalize(Min-Max) the image.
        #image[:, :, 2] = normalize(V_channel)
        image.astype(np.float32)
        X_train[i] = image

    for i in range(len(X_Validate)):
        image = X_Validate[i]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        V_channel = image[:, :, 2]
        image[:, :, 2] = standardization(V_channel)
        # Normalize(Min-Max) the image.
        #image[:, :, 2] = normalize(V_channel)
        image.astype(np.float32)
        X_Validate[i] = image

    for i in range(len(X_test)):
        image = X_test[i]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        V_channel = image[:, :, 2]
        image[:, :, 2] = standardization(V_channel)
        # Normalize(Min-Max) the image.
        #image[:, :, 2] = normalize(V_channel)
        image.astype(np.float32)
        X_test[i] = image

def global_local_contrast_normalization(image):
    # Convert the image to yuv colorspace and do
    # local and global normalization on the brightness channel.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess_data():
    """"Data pre processing and augmentation."""
    global X_train
    global y_train
    global X_Validate
    global y_validate
    augment_data()
    one_hot_encode()
    shuffle_data()
    #split_data()
    X_train, X_Validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2,
                                                                                    random_state=0)
    # normalize_data()

def next_batch(batch_size):
    """"Generator to provide next batch of data."""
    global X_train
    global y_train
    batch_start = 0
    batch_end = 0
    data_len = len(y_train)
    batch_len = ceil(data_len/batch_size)
    for itr in range(batch_len):
        batch_start = batch_end
        batch_end = batch_end + batch_size
        yield (X_train[batch_start:batch_end], y_train[batch_start:batch_end])


def oneCNN_network(x, prob):
    """One CNN network for traffic sign classification."""
    mu = 0
    sigma = 0.1
    # input_images = tf.image.resize_images(x, [48, 48])

    # layer 1 5x5 kernel and stride 1.
    layer_1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 64), mean=mu, stddev=sigma))
    layer_1_b = tf.Variable(tf.zeros(64))
    layer_1_conv = tf.nn.conv2d(x, layer_1_w, strides=[1, 1, 1, 1], padding='SAME') + layer_1_b
    # Maxpool kernel 2x2 stride 2
    layer_1_conv = tf.nn.max_pool(layer_1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Two branches: fully connected and convolutional.
    # branch_1_data = tf.reshape(layer_1_conv, [-1])
    branch_1_data = flatten(layer_1_conv)
    weight_1_fc = tf.Variable(tf.truncated_normal(shape=(int(branch_1_data.get_shape()[1]), 96), mean=mu, stddev=sigma))
    bias_1_fc = tf.Variable(tf.zeros(96))
    branch_1_output = tf.add(tf.matmul(branch_1_data, weight_1_fc), bias_1_fc)
    branch_1_output = tf.nn.dropout(branch_1_output, prob)

    layer_2_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean=mu, stddev=sigma))
    layer_2_b = tf.Variable(tf.zeros(128))
    layer_2_conv = tf.nn.conv2d(layer_1_conv, layer_2_w, strides=[1, 1, 1, 1], padding='SAME') + layer_2_b
    layer_2_conv = tf.nn.max_pool(layer_2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Two further branches.
    branch_2_data = flatten(layer_2_conv)
    weight_2_fc = tf.Variable(tf.truncated_normal(shape=(int(branch_2_data.get_shape()[1]), 96), mean=mu, stddev=sigma))
    bias_2_fc = tf.Variable(tf.zeros(96))
    branch_2_output = tf.add(tf.matmul(branch_2_data, weight_2_fc), bias_2_fc)
    branch_2_output = tf.nn.dropout(branch_2_output, prob)

    layer_3_w = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 64), mean=mu, stddev=sigma))
    layer_3_b = tf.Variable(tf.zeros(64))
    layer_3_conv = tf.nn.conv2d(layer_2_conv, layer_3_w, strides=[1, 1, 1, 1], padding='SAME') + layer_3_b
    layer_3_conv = tf.nn.max_pool(layer_3_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    branch_3_data = flatten(layer_3_conv)
    weight_3_fc = tf.Variable(tf.truncated_normal(shape=(int(branch_3_data.get_shape()[1]), 96), mean=mu, stddev=sigma))
    bias_3_fc = tf.Variable(tf.zeros(96))
    branch_3_output = tf.add(tf.matmul(branch_3_data, weight_3_fc), bias_3_fc)
    branch_3_output = tf.nn.dropout(branch_3_output, prob)

    # Concat the three branches.
    branch_join_data = tf.concat(1, [branch_1_output, branch_2_output, branch_3_output])
    branch_join_data = tf.nn.relu(branch_join_data)

    # Two fully connected layers with relu.
    layer_4_fc_w = tf.Variable(tf.truncated_normal(shape=(int(branch_join_data.get_shape()[1]), 100), mean=mu, stddev=sigma))
    layer_4_fc_b = tf.Variable(tf.zeros(100))
    layer_4_fc_output = tf.add(tf.matmul(branch_join_data, layer_4_fc_w), layer_4_fc_b)
    layer_4_fc_output = tf.nn.relu(layer_4_fc_output)
    layer_4_fc_output = tf.nn.dropout(layer_4_fc_output, prob)

    layer_5_fc_w = tf.Variable(tf.truncated_normal(shape=(int(layer_4_fc_output.get_shape()[1]), 43), mean=mu, stddev=sigma))
    layer_5_fc_b = tf.Variable(tf.zeros(43))
    layer_5_output = tf.add(tf.matmul(layer_4_fc_output, layer_5_fc_w), layer_5_fc_b)

    return layer_5_output


def network(x, prob):
    """"Implement the training chain."""
    # Layer 1.
    layer_1_conv = tf.nn.conv2d(x, weights['w1'], strides=[1, 1, 1, 1],padding='SAME')
    layer_1_conv = tf.nn.bias_add(layer_1_conv, biases_s['b1'])
    layer_1_conv = tf.nn.relu(layer_1_conv)
    layer_1_conv = tf.nn.max_pool(layer_1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer_1_conv = tf.contrib.layers.batch_norm(layer_1_conv, is_training=is_training, trainable=True)

    # Layer 2
    layer_2_conv = tf.nn.conv2d(layer_1_conv, weights['w2'], strides=[1, 1, 1, 1], padding='SAME')
    layer_2_conv = tf.nn.bias_add(layer_2_conv, biases_s['b2'])
    layer_2_conv = tf.nn.relu(layer_2_conv)
    layer_2_conv = tf.nn.max_pool(layer_2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer_2_conv = tf.contrib.layers.batch_norm(layer_2_conv, is_training=is_training, trainable=True)

    # layer 3 - First fully connected.
    fc0 = flatten(layer_2_conv)
    fc0 = tf.nn.dropout(fc0, 0.8)
    layer_3_conv = tf.add(tf.matmul(fc0, weights['w3']), biases_s['b3'])
    layer_3_conv = tf.nn.relu(layer_3_conv)
    layer_3_conv = tf.nn.dropout(layer_3_conv, prob)

    # layer_4 - Second fully connected.
    layer_4_conv = tf.add(tf.matmul(layer_3_conv, weights['w4']), biases_s['b4'])
    layer_4_conv = tf.nn.relu(layer_4_conv)
    layer_4_conv = tf.nn.dropout(layer_4_conv, prob)

    #layer_5 - Third fully connected.
    layer_5_conv = tf.add(tf.matmul(layer_4_conv, weights['w5']), biases_s['b5'])
    layer_5_conv = tf.nn.relu(layer_5_conv)
    # layer_5_conv = tf.nn.dropout(layer_5_conv, prob)
    logits = tf.add(tf.matmul(layer_5_conv, weights['w6']), biases_s['b6'])
    return logits


def validate(batch_size, accuracy, x, y, prob):
    """"Evaluate the network on validation data."""
    global X_Validate
    global y_validate
    global is_training
    total_accuracy = 0
    is_training = False
    validation_size = len(y_validate)
    assert len(X_Validate) == validation_size

    sess = tf.get_default_session()
    for offset in range(0, validation_size, batch_size):
        batch_x, batch_y = X_Validate[offset: offset + batch_size], y_validate[offset: offset + batch_size]
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, prob:0.5})
        total_accuracy += (acc * batch_size)
    is_training = True # Enable it back for training.
    return total_accuracy/validation_size


def test(batch_size, accuracy, x, y, prob):
    """"Check the network on test data."""
    global X_test
    global y_test
    global is_training
    total_accuracy = 0
    is_training = False
    test_size = len(y_test)
    assert len(X_test) == test_size
    sess = tf.get_default_session()
    for offset in range(0, test_size, batch_size):
        batch_x, batch_y = X_test[offset: offset + batch_size], y_test[offset: offset + batch_size]
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, prob: 0.5})
        total_accuracy += (acc * batch_size)
    return total_accuracy / test_size


def train(epochs, batch_size, learning_rate):
    """Train the network."""
    global x_input
    global y_labels
    global is_training
    global model
    train_acc = {"acc":[], "iter":[]}
    valid_acc = {"acc":[], "iter":[]}
    select_network = 0
    is_training = True
    # Pick the network to use for training.
    if select_network == 0 :
        model = network(x_input, prob)
    else:
        model = oneCNN_network(x_input, prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Define accuracy function.
    accuracy_operation = tf.equal(tf.argmax(model, 1), tf.argmax(y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(accuracy_operation, tf.float32))
    batch_count = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for batchset in next_batch(batch_size):
                batch_count += 1
                sess.run(optimizer, feed_dict={x_input:batchset[0], y_labels:batchset[1], prob:0.5})
                if batch_count%batch_size == 0:
                    acc = sess.run(accuracy, feed_dict={x_input:batchset[0], y_labels:batchset[1], prob:0.5})
                    train_acc["acc"].append(acc)
                    train_acc["iter"].append(batch_count)
            acc = validate(batch_size, accuracy, x_input, y_labels, prob)
            valid_acc["acc"].append(acc)
            valid_acc["iter"].append(batch_count)
            print('Epoch: {:>2} - Batch:{:>5} - Accuracy: {:>5.4f}'.format(epoch, batch_count, acc))
        acc = test(batch_size, accuracy, x_input, y_labels, prob)

        print('Test Accuracy -: {:>5.4f}'.format(acc))
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved to : ", save_path)
        plot_acc(train_acc, valid_acc)

def plot_acc(train, valid):
    plt.plot(train["iter"], train["acc"], 'b')
    plt.plot(valid["iter"], valid["acc"], 'r')
    plt.show()

def detect_image(files):
    """To test the model on unknown additional data."""
    global model
    global x_input
    image_ids = list()
    model = network(x_input, 0.5)
    #Resize the image.
    arg_max = tf.argmax(model, 1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model.ckpt")
        for image in files:
            img = cv2.imread(image)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(img.shape)
            img = np.array([img])
            image_id = sess.run(arg_max, feed_dict={x_input: img})
            image_ids.append(image_id[0])
    return image_ids


def process_collected_image(files):
    image_ids = detect_image(files)
    # Plot the image and id.
    for index, image in enumerate(files):
        img = plt.imread(image)
        plt.imshow(img)
        plt.xlabel(content[image_ids[index]][0])
        plt.show()

#normalize_data()
visualize_data()
# preprocess_data()
# train(25, 128, 0.0003) # Train for 50 epochs with batch size of 128 and training rate of 0.0001
# Test on additional data.
def additional_data_test():
    additional_images = list()
    additional_images.append('../additional_data/branching_road.jpg')
    additional_images.append('../additional_data/pedestrian.jpg')
    additional_images.append('../additional_data/reverse_curve.JPG')
    additional_images.append('../additional_data/No_u_turn.JPG')
    additional_images.append('../additional_data/stop_ahead.JPG')
    additional_images.append('../additional_data/turn.jpg')
    additional_images.append('../additional_data/twisty_road.jpg')
    process_collected_image(additional_images)

additional_data_test()