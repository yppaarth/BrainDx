from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
        self.steps_per_epoch = None
        self.validation_steps = None

    def get_base_model(self):
        print(f"Loading base model from {self.config.updated_base_model_path}")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        print("Base model loaded and compiled successfully.")

        # Auto initialize metrics with dummy input/output
        output_units = self.model.output_shape[-1]
        dummy_input = tf.random.uniform((1, *self.config.params_image_size))
        dummy_output = tf.one_hot([0], depth=output_units)
        self.model.train_on_batch(dummy_input, dummy_output)
        print(f"Metrics initialized: {self.model.metrics_names}")

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        print("Creating validation data generator...")
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        print("Creating training data generator...")
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        print("Data generators created successfully.")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        print(f"Saving trained model to {path}")
        model.save(path)
        print("Model saved successfully.")

    def train(self, callback_list: list):
        assert self.model is not None, "Model has not been initialized."
        assert self.train_generator is not None, "Training data generator is not set up."
        assert self.valid_generator is not None, "Validation data generator is not set up."

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Optionally compute class weights
        class_weights_dict = None
        if getattr(self.config, "use_class_weights", False):
            print("Computing class weights...")
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.train_generator.classes),
                y=self.train_generator.classes
            )
            class_weights_dict = dict(enumerate(class_weights))
            print(f"Class weights: {class_weights_dict}")

        print(f"Starting training for {self.config.params_epochs} epochs...")
        print(f"Steps per epoch: {self.steps_per_epoch}, Validation steps: {self.validation_steps}")

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
            class_weight=class_weights_dict
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

        print("Training completed.")

        # Optional Confusion Matrix and Classification Report
        if getattr(self.config, "generate_confusion_matrix", False):
            print("Generating classification report and confusion matrix...")
            val_preds = self.model.predict(self.valid_generator)
            pred_labels = np.argmax(val_preds, axis=1)
            true_labels = self.valid_generator.classes

            print(classification_report(true_labels, pred_labels))
            cm = confusion_matrix(true_labels, pred_labels)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()

        return history
