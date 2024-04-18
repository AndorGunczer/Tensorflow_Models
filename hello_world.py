import tensorflow as tf

def main():
    print("tensorflow version: " + tf.__version__)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255, x_test/255

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    # What do the different layers mean?

    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    # Why is numpy() used here?
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # What are logits?

    loss_fn(y_train[:1], predictions).numpy()
    model.compile(  optimizer="adam",
                    loss=loss_fn,
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


    

if __name__ == "__main__":
    main()