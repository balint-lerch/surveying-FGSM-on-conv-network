import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import keras
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K

#Model loading and prediction
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
print(test_images[0])
test_images = test_images.astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

clothes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.load_model("2Conv2D", compile=False)
predictions = model.predict(test_images)

#Adversarial example generating
epsilon = 0.07
n_classes = 10

x_adv = test_images
sess = K.get_session()

initial_class = np.argmax(model.predict(test_images), axis=1)
target = K.one_hot(initial_class, n_classes)

#Getting the loss function of the output and taking the gradient of it
loss = K.categorical_crossentropy(target, model.output)
grads = K.gradients(loss, model.input)

#FGSM method. Perturbating the original image with the epsilon * sign of gradient
delta = K.sign(grads[0])
x_adv = x_adv + epsilon * delta
x_adv = sess.run(x_adv, feed_dict={model.input: test_images})
preds = model.predict(x_adv)

#Showing the results:

#testing 19, 3111, 15, 6542, 49, 42

#Showing the original/perturbated pairs
plt.figure()
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(x_adv[0], cmap=plt.cm.binary)
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(test_images[17], cmap=plt.cm.binary)
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(x_adv[17], cmap=plt.cm.binary)
plt.grid(False)
plt.show()

#Helper functions to easily display plots
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(clothes[predicted_label],
                                         100 * np.max(predictions_array),
                                         clothes[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.xticks(x_ticks, clothes, rotation="vertical")
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

PICT = 4
PICT2 = 17

#Displaying the predictions of the original/perturbated pairs
plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plot_image(PICT, predictions[PICT], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(PICT, predictions[PICT], test_labels)
plt.tight_layout()
plt.show()


#19 42 49 6542

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plot_image(PICT, preds[PICT], test_labels, x_adv)
plt.subplot(1, 2, 2)
plot_value_array(PICT, preds[PICT], test_labels)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plot_image(PICT2, predictions[PICT2], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(PICT2, predictions[PICT2], test_labels)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plot_image(PICT2, preds[PICT2], test_labels, x_adv)
plt.subplot(1, 2, 2)
plot_value_array(PICT2, preds[PICT2], test_labels)
plt.tight_layout()
plt.show()

#Displaying the outputs of the convolutional and pooling layers for the original/perturbated pairs
f, axarr = plt.subplots(4, 4)
FIRST_IMAGE = 0
FIRST_ADV_IMAGE = 0
SECOND_IMAGE = 6542
SECOND_ADV_IMAGE = 6542
#I'm using a 64 filtered convolutional layer so with this variable we can switch between those filters
CONVOLUTION_NUMBER = 1 #3 23
layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
for x in range(4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0, x].grid(False)
    f2 = activation_model.predict(x_adv[FIRST_ADV_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[1, x].grid(False)
    f3 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[2, x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[2, x].grid(False)
    f4 = activation_model.predict(x_adv[SECOND_ADV_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[3, x].imshow(f4[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[3, x].grid(False)
plt.show()
