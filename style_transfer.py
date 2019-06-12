# Import necessary Packages
import numpy as np
from PIL import Image
import cv2
import tkinter
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b


# Global Variables

ITERATION = 5

IMG_WIDTH, IMG_HEIGHT = 300, 300
CHANNELS = 3

CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5

TOTAL_VARIATION = 0.995
TOTAL_LOSS_FACTOR = 1.25

# Define paths for input and style images

input_path = 'input_images/Beagle.jpg'

style_path = 'style_images/1.jpg'


def process_input(input_path):

    # Pre-process input image
    input_image = cv2.imread(input_path)
    input_image = cv2.resize(input_image, (IMG_WIDTH, IMG_HEIGHT))
    input_array = np.float32(input_image)
    input_array[:, :, 0] -= 123.68
    input_array[:, :, 1] -= 116.779
    input_array[:, :, 2] -= 103.939
    input_array = np.expand_dims(input_array, axis=0)
    return input_array

def process_style(style_path):
    # Preprocess Style image
    style_image = cv2.imread(style_path)
    style_image = cv2.resize(style_image, (IMG_WIDTH, IMG_HEIGHT))
    style_array = np.float32(style_image)
    style_array[:, :, 0] -= 123.68
    style_array[:, :, 1] -= 116.779
    style_array[:, :, 2] -= 103.939
    style_array = np.expand_dims(style_array, axis=0)
    return style_array

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMG_HEIGHT * IMG_WIDTH
    return backend.sum(backend.square(style - combination)) / (4 * (CHANNELS ** 2) * (size ** 2))

def total_variation_loss(x):
    a = backend.square(x[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - x[:, 1:, :IMG_WIDTH-1, :])
    b = backend.square(x[:, :IMG_HEIGHT-1, :IMG_WIDTH-1, :] - x[:, :IMG_HEIGHT-1, 1:, :])
    return backend.sum(backend.pow(a + b, TOTAL_LOSS_FACTOR))

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    outs = backend.function([combination], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients


class Evaluation:

    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients


##########################################################

if __name__ == '__main__':

    input_array = process_input(input_path)
    style_array = process_style(style_path)
    
    # Create a VGG16 model
    input_image = backend.variable(input_array)
    style_image = backend.variable(style_array)
    combination = backend.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))

    input_tensor = backend.concatenate([input_image, style_image, combination], axis=0)
    model = VGG16(input_tensor=input_tensor, include_top=False)
    model.summary()

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    # Choose Preferred convolutional blocks
    content_layer = "block2_conv2"

    style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]

    layer_features = layers[content_layer]

    content_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]

    loss = backend.variable(0)
    loss += CONTENT_WEIGHT * content_loss(content_image_features, combination_features)


    for layer_name in style_layers:
        layer_features = layers[layer_name]
        
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        
        content_style_loss = style_loss(style_features, combination_features)
        loss += (STYLE_WEIGHT / len(style_layers)) * content_style_loss


    loss += TOTAL_VARIATION * total_variation_loss(combination)

    outputs = [loss]
    outputs += backend.gradients(loss, combination)

    evaluator = Evaluation()
    x = np.random.uniform(0, 255, (1, IMG_HEIGHT, IMG_WIDTH, 3)) - 128

    for i in range(ITERATION):
        x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
        print("Iteration %d completed with loss %d" % (i, loss))

    x = x.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype("uint8")

    cv2.imwrite('stylized.jpg', x)