import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocess input image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize if not already done in model

        # Predict class
        prediction_idx = np.argmax(model.predict(test_image), axis=1)[0]
        
        # Map prediction index to class name
        class_labels = {
            0: "NonDemented",
            1: "MildDemented",
            2: "VeryMildDemented",
            3: "ModerateDemented"
        }
        
        prediction = class_labels.get(prediction_idx, "Unknown")
        return [{"image": prediction}]
