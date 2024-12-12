from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0


# model with 2 layer of RNN Block

model = Sequential()
model.add(
    SimpleRNN(units=64, input_shape=(28, 28), activation="relu", return_sequences=True)
)
model.add(SimpleRNN(units=32, activation="relu", return_sequences=True))
model.add(SimpleRNN(units=16, activation="relu"))
model.add(Dense(10, activation="softmax"))


optimizer = Adam(learning_rate=0.001)


model.compile(
    optimizer=optimizer, loss=["sparse_categorical_crossentropy"], metrics=["accuracy"]
)


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)
