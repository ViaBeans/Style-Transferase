import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from skimage.transform import resize
import imageio
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)
tf.compat.v1.disable_eager_execution()

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = sys.argv[1]
STYLE_IMG_PATH = sys.argv[2]
CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.025
STYLE_WEIGHT = 1.0
TOTAL_WEIGHT = 8.5e-5

TRANSFER_ROUNDS = 20




def deprocessImage(img):
    # subtracting mean RGB values (got it from google) and reversing order of R, G and B.
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def styleLoss(style, gen):
    filter_num = 3
    k_sum = K.sum(K.square(gramMatrix(style) - gramMatrix(gen)))
    return k_sum / (4.0 * (filter_num ^ 2) * ((STYLE_IMG_H * STYLE_IMG_W) ^ 2))


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    a = tf.square(
        x[:, : CONTENT_IMG_W - 1, : CONTENT_IMG_H - 1, :] -
        x[:, 1:, : CONTENT_IMG_H - 1, :]
    )
    b = tf.square(
        x[:, : CONTENT_IMG_W - 1, : CONTENT_IMG_H - 1, :] -
        x[:, : CONTENT_IMG_W - 1, 1:, :]
    )
    # point of this function is to serve as a variation loss function. Keeps the generated image coherent (style transfer applies equally across the image). https://keras.io/examples/generative/neural_style_transfer/
    return K.sum(K.pow(a+b, 1.25))


def loss_grads_ret(x, func):
    x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_H, 3))
    outs = func([x])
    loss = outs[0]
    grad = outs[1].flatten().astype('float64')
    return loss, grad


class eval(object):
    def __init__(self, func):
        self.loss_arr = None
        self.grad_arr = None
        self.func = func

    def loss(self, x):
        loss_val, grad_val = loss_grads_ret(x, self.func)
        self.loss_arr = loss_val
        self.grad_arr = grad_val
        return self.loss_arr

    def grads(self, x):
        return self.grad_arr


def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = resize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# Constructing model, custom loss function, style transfer with gradient descent, and save image


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))

    inputTensor = K.concatenate(
        [contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(weights="imagenet",
                        include_top=False, input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])

    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1",
                       "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)
    print("   Calculating style loss.")

    for layerName in styleLayerNames:
        l_features = outputDict[layerName]
        sr_features = l_features[1, :, :, :]
        comb_features = l_features[2, :, :, :]

        s_loss = styleLoss(sr_features, comb_features)
        loss += (STYLE_WEIGHT / len(styleLayerNames)) * s_loss

    loss += totalLoss(genTensor) * TOTAL_WEIGHT
    out = [loss]
    gradients = K.gradients(loss, genTensor)[0]
    if isinstance(gradients, (list, tuple)):
        out += gradients
    else:
        out.append(gradients)

    func = K.function([genTensor], out)
    r_data = tData
    eval_obj = eval(func)

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        r_data, tLoss, i = fmin_l_bfgs_b(
            eval_obj.loss, r_data.flatten(), fprime=eval_obj.grads, maxfun=25)
        print("      Loss: %f." % tLoss)
        img = deprocessImage(r_data.copy())
        f_path = "transferred_im.png"
        imageio.imwrite(f_path, img)
        print("      Image saved to \"%s\"." % f_path)
    print("   Transfer complete.")


def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()
