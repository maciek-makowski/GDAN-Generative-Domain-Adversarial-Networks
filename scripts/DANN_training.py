import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf 
import numpy as np 
import keras
from keras.layers import Conv1D, Dense, Flatten, InputLayer
from tensorflow.keras import models
from sklearn.utils import shuffle


class DANN(models.Model):
    def __init__(self, no_features):
        super(DANN, self).__init__()
        input_shape = (no_features, 1)
        self.feature_extractor = self.build_feature_extractor(input_shape)[0]
        feature_extractor_output = self.build_feature_extractor(input_shape)[1]
        self.label_classifier = self.build_label_classifier(feature_extractor_output)
        self.domain_classifier = self.build_domain_classifier(feature_extractor_output)

    
    def build_feature_extractor(self, input_shape):
        model = keras.models.Sequential()
        model.add(Conv1D(32, kernel_size=5, input_shape = input_shape, activation='relu'))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(Flatten())
        output_shape = model.output_shape[1]
        return model, output_shape
    
    def build_label_classifier(self, input_shape):
        model = tf.keras.Sequential()
        model.add(InputLayer(shape = (input_shape,)))
        model.add(Dense(2*input_shape))
        model.add(Dense(64))
        model.add(Dense(2, activation='softmax'))
        return model
    
    def build_domain_classifier(self, input_shape):
        model= tf.keras.Sequential()
        model.add(InputLayer(shape = (input_shape,)))
        model.add(Dense(2*input_shape))
        model.add(Dense(64))
        model.add(Dense(2, activation='softmax'))
        return model 

    def call(self, inputs, training=None, **kwargs):
        features = self.feature_extractor(inputs)
        features_grl = self.grl(features)
        label_pred = self.label_classifier(features)
        domain_pred = self.domain_classifier(features_grl)
        return label_pred, domain_pred


def train_dann_model(model, source_data, source_labels, target_data, target_labels, num_epochs=10, batch_size=32, learning_rate=0.001, lambda_value=1.0):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Prepare domain labels (0 for source domain, 1 for target domain)
    source_domain_labels = np.zeros((len(source_data), 1))
    target_domain_labels = np.ones((len(target_data), 1))

    # Concatenate source and target data
    combined_data = np.concatenate((source_data, target_data), axis=0)
    combined_labels = np.concatenate((source_labels, target_labels), axis=0)
    combined_domain_labels = np.concatenate((source_domain_labels, target_domain_labels), axis=0)

    # Shuffle combined data
    combined_data, combined_labels, combined_domain_labels = shuffle(combined_data, combined_labels, combined_domain_labels)

     # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step in range(0, len(combined_data), batch_size):
            batch_data = combined_data[step:step+batch_size]
            batch_labels = combined_labels[step:step+batch_size]
            batch_domain_labels = combined_domain_labels[step:step+batch_size]

            # Perform forward pass once
            with tf.GradientTape() as tape:
                features = model.feature_extractor(batch_data, training=True)
                label_pred = model.label_classifier(features)
                domain_pred = model.domain_classifier(features)

                # Compute label loss
                label_loss = tf.keras.losses.categorical_crossentropy(batch_labels, label_pred)

                # Compute domain loss
                domain_loss = tf.keras.losses.categorical_crossentropy(batch_domain_labels, domain_pred)

            # Compute gradients for feature extractor with respect to label loss
            feature_gradients_label = tape.gradient(label_loss, model.feature_extractor.trainable_variables)

            # Compute gradients for feature extractor with respect to domain loss
            feature_gradients_domain = tape.gradient(domain_loss, model.feature_extractor.trainable_variables)

            # Update weights for feature extractor
            for i, var in enumerate(model.feature_extractor.trainable_variables):
                var.assign_sub(feature_gradients_label[i] - lambda_value * feature_gradients_domain[i]) ## CHECK IF LR is needed or not

            # Compute gradients for label classifier
            label_gradients = tape.gradient(label_loss, model.label_classifier.trainable_variables)
 
            # Compute gradients for domain classifier
            domain_gradients = tape.gradient(domain_loss, model.domain_classifier.trainable_variables)

            # Update weights for label classifier
            model.label_classifier.optimizer.apply_gradients(zip(label_gradients, model.label_classifier.trainable_variables))

            # Update weights for domain classifier
            model.domain_classifier.optimizer.apply_gradients(zip(domain_gradients, model.domain_classifier.trainable_variables))
            
            # Print training progress
            print(f"Step {step+1}/{len(combined_data)} - Label Loss: {label_loss}, Domain Loss: {domain_loss}")

    print("Training finished.")

x = DANN(11)