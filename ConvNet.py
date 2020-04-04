from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255
train_labels = to_categorical(train_labels)

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255
test_labels = to_categorical(test_labels)

network = Sequential()
network.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
network.add(Conv2D(64, (3, 3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))
network.add(Flatten())
network.add(Dense(128, activation='relu'))
network.add(Dropout(0.5))
network.add(Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.1)

results = network.evaluate(test_images, test_labels)
print("\n\n")
print(results)
