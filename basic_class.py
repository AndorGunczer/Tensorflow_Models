import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def displayAnImage():
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def displayImages():
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

def main():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images, test_images = train_images/255, test_images/255

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten an 28x28 matrix
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    probability_model = tf.keras.models.Sequential([model, tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    




if __name__ == "__main__":
    main()