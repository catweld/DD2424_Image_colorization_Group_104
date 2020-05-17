import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

"""
This files uses the pre-trained model Inception_v3 which is a CNN used for image analysis and object detection.
It is trained on the ImageNet data set and has state of the art performance.
"""


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # This line allows the network to use the GPU VRAM uncapped. !!! NEED THIS LINE FOR NETWORK TO RUN !!!
        for idx, g in enumerate(gpus):
            tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[idx], True)
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)


def main():
    save_results_as = "Dog_random"
    image_path = "images/Dog_random/"
    images = os.listdir(image_path)
    classifications = []
    i = 0
    for image in tqdm(images):
        classifications.append((image, classify_images(image_path+image)))

    save_to_file(save_results_as+"-classifications", classifications)


def classify_images(image_full_path):
    """
    Input a image and it will return the top 3 classes that the networks thinks the picture is.
    The classes is based on the ImageNet and is it is 1k classes in total.
    :return: void
    """

    # Load the desired image
    img = image.load_img(image_full_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = InceptionV3(weights="imagenet")
    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    tf.keras.backend.clear_session()
    print('Predicted:', decode_predictions(preds, top=3)[0])
    return decode_predictions(preds, top=3)[0]


def save_to_file(file_name, classifications):
    with open('classification_results/'+ file_name+".txt", 'w') as f:
        f.write("This is the classifications results")
        image_batch = 0
        for item in classifications:
            if image_batch % len(item[1]) == 0:
                f.write("\n---------- New image under ------------------")
            f.write("\nImage name: " + str(item[0]) + ", Top classifications: " + str(item[1]))
            image_batch += 1


if __name__ == "__main__":
    main()
