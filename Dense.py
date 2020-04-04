from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
train_labels = to_categorical(train_labels)

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
test_labels = to_categorical(test_labels)

sgd = SGD(lr=0.125, momentum=0.9, decay=0.0, nesterov=False)
network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(train_images, train_labels, epochs=20, batch_size=250, validation_split=0.1, verbose=2)

results = network.evaluate(test_images, test_labels)
print("\n\n")
print(results)
