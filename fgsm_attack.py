import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import time
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def create_victim_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

victim_model = create_victim_model()

victim_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

start_train_time = time.time()
history = victim_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"Training Time: {train_time} seconds")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png') #save the figure.
plt.show()

test_loss, test_acc = victim_model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

def generate_adversarial_pattern(image, label, model, epsilon=0.1):
    image = tf.convert_to_tensor(image)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = losses.categorical_crossentropy(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adversarial_pattern = signed_grad * epsilon
    return adversarial_pattern

def generate_adversarial_example(image, label, model, epsilon=0.1):
    adversarial_pattern = generate_adversarial_pattern(image, label, model, epsilon)
    adversarial_example = image + adversarial_pattern
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1) #ensure image values remain between 0 and 1
    return adversarial_example

def evaluate_attack(model, images, labels, epsilon=0.1):
    correct_predictions = 0
    attack_success = 0
    total_samples = len(images)

    for i in range(total_samples):
        image = images[i:i+1]
        label = labels[i:i+1]

        original_prediction = np.argmax(model.predict(image), axis=1)[0]
        if np.argmax(label) == original_prediction:
            correct_predictions += 1

            adversarial_example = generate_adversarial_example(image, label, model, epsilon)
            adversarial_prediction = np.argmax(model.predict(adversarial_example), axis=1)[0]

            if adversarial_prediction != np.argmax(label):
                attack_success += 1
    if correct_predictions==0:
        return 0
    success_rate = attack_success / correct_predictions
    return success_rate

start_attack_time = time.time()
epsilon = 0.1  # Adjust epsilon for attack strength
success_rate = evaluate_attack(victim_model, test_images, test_labels, epsilon)
end_attack_time = time.time()
attack_time = end_attack_time - start_attack_time
print(f"Attack Time: {attack_time} seconds")
print(f'FGSM Attack Success Rate (Epsilon={epsilon}): {success_rate}')

num_samples = 10
fig, axes = plt.subplots(num_samples, 2, figsize=(8, 20))
for i in range(num_samples):
    image = test_images[i:i+1]
    label = test_labels[i:i+1]
    adversarial_example = generate_adversarial_example(image, label, victim_model, epsilon)

    axes[i, 0].imshow(image[0, :, :, 0], cmap='gray')
    axes[i, 0].set_title(f"Clean: {np.argmax(label)}")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(adversarial_example[0, :, :, 0], cmap='gray')
    axes[i, 1].set_title(f"Adv: {np.argmax(victim_model.predict(adversarial_example))}")
    axes[i, 1].axis('off')
plt.savefig("adversarial_examples.png")
plt.show()