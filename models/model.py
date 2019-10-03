from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import abc

import numpy as np
import sys, os

# import os.path
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

bundle_dir = None
if getattr(sys, "frozen", False):
    bundle_dir = sys._MEIPASS
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class MaskingModel(metaclass=abc.ABCMeta):
    """abstract class to guarantee children will
    have predict_mask method"""

    @abc.abstractmethod
    def predict_mask(self, image):
        """abstract method, to be implemented by child classes"""
        pass


class Unet(MaskingModel):
    """Unet class to manage the loding of model and weights and
    predictive use"""

    def __init__(self):
        """Class constructor get json model and h5 weigths and load model"""

        if bundle_dir:
            weight_path = os.path.join(bundle_dir, "models/weights/unet_weights.h5")
            model_path = os.path.join(bundle_dir, "models/json_models/unet_model.json")
        else:
            weight_path = "models/weights/unet_weights.h5"
            model_path = "models/json_models/unet_model.json"

        json_file = open(model_path, "r")
        json_model = json_file.read()
        json_file.close()

        self.unet_model = model_from_json(json_model)
        self.unet_model.load_weights(weight_path)

    def __chooseMainComponent(self, image):
        """ChooseMainComponent function to only keep the
        largest component of a prediction, removes unwanted
        artifacts"""

        image = image.astype("uint8")
        # 3D image backbone
        new_image = np.zeros(image.shape)

        # go slice by slice of a prediction,
        # and find best component
        for i in range(image.shape[0]):
            image_slice = image[i, :, :, :]
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                image_slice, connectivity=4
            )
            sizes = stats[:, -1]

            # set the label that shows the largest compont
            max_label = 1

            # only one component
            if len(sizes) < 3:
                return image

            max_size = sizes[1]

            # get the largest component
            for j in range(2, nb_components):
                if sizes[j] > max_size:
                    max_label = j
                    max_size = sizes[j]

            # 2D image slice, keep only largest component
            new_slice = np.zeros(output.shape)
            new_slice[output == max_label] = 1
            new_slice = new_slice[..., np.newaxis]

            # append 2D image to 3D image
            new_image[i, :, :, :] = new_slice

        return new_image

    def __getGenerator(self, image, bs=1):
        """getGenerator Returns generator that will be used for
        prdicting, it takes a single 3D image and returns a generator
        of its slices"""

        # rescale data to its trained mode
        image_datagen = ImageDataGenerator(rescale=1.0 / 255)
        image_datagen.fit(image, augment=True)
        image_generator = image_datagen.flow(x=image, batch_size=bs, shuffle=False)

        return image_generator

    def predict_mask(self, image):
        """predict_mask creates a prediction for a whole 3D image"""
        image_gen = self.__getGenerator(image)
        mask = self.unet_model.predict_generator(image_gen, steps=len(image))
        # only keep pixels with more than 0.5% probability of being brain
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        mask = self.__chooseMainComponent(mask)

        return mask
