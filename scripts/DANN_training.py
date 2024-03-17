import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf 
import numpy as np 
import keras
from keras.layers import Conv1D, Dense, Flatten, InputLayer, AveragePooling1D, Reshape, BatchNormalization, MaxPooling1D, Normalization, Dropout, Conv1DTranspose
from tensorflow.keras import models
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class DANN(models.Model):
    def __init__(self, no_features, regularization_coeff=0.01):
        super(DANN, self).__init__()
        self.regularization_coeff = regularization_coeff
        input_shape = (no_features, 1)
        #input_shape = (no_features,)
        self.feature_extractor, feature_extractor_output = self.build_feature_extractor(input_shape)
        print(feature_extractor_output)
        self.label_classifier = self.build_label_classifier(feature_extractor_output)
        self.domain_classifier = self.build_domain_classifier(feature_extractor_output)

    
    def build_feature_extractor(self, input_shape):
        model = keras.models.Sequential()
        model.add(InputLayer(shape = input_shape))
        model.add(Conv1DTranspose(32, kernel_size=6, strides=2, input_shape = input_shape, activation='relu'))# kernel_regularizer=l2(self.regularization_coeff)
        model.add(BatchNormalization())
        model.add(Conv1DTranspose(64, kernel_size=8, strides=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(32, kernel_size=6, strides=14, activation='relu'))
        model.add(Flatten())
        model.add(Reshape((model.output_shape[1], 1)))
        model.add(AveragePooling1D(pool_size=37, strides=20))
        #model.add(MaxPooling1D(pool_size=37, strides=20))
        model.add(Flatten())
        model.add(Dense(11, input_shape = input_shape))
        model.add(Normalization(mean = np.zeros(11), variance= np.ones(11)))
        model.summary()
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer)
        ### Tune pool_size and strides according to output = (len(flatten) + 1 - pool_size) / stride
        output_shape = model.output_shape[1]
        return model, output_shape
    
    def build_label_classifier(self, input_shape):
        model = tf.keras.Sequential()
        model.add(InputLayer(shape = (input_shape,)))
        model.add(Dense(256,activation= 'relu')) # kernel_regularizer=l2(self.regularization_coeff)
        model.add(Dense(512,activation='relu')) # , kernel_regularizer=l2(self.regularization_coeff)
        model.add(Dense(64,activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model
    
    def build_domain_classifier(self, input_shape):
        model= tf.keras.Sequential()
        model.add(InputLayer(shape = (input_shape,)))
        model.add(Dense(512, activation='relu')) # kernel_regularizer=l2(self.regularization_coeff)
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))# kernel_regularizer=l2(self.regularization_coeff)
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        optimizer = tf.keras.optimizers.Adam()  # Define optimizer
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])  # Compile the model
        return model 

    def call(self, inputs, training=None, **kwargs):
        features = self.feature_extractor(inputs)
        features_grl = self.grl(features)
        label_pred = self.label_classifier(features)
        domain_pred = self.domain_classifier(features_grl)
        return label_pred, domain_pred


def train_dann_model(model, source_data, source_labels, target_data, target_labels, num_epochs=10, batch_size=512, learning_rate=1e-5, lambda_value=0.1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Prepare domain labels (0 for source domain, 1 for target domain)
    source_domain_labels = np.zeros((len(source_data), 1))
    target_domain_labels = np.ones((len(target_data), 1))

    # Concatenate source and target data
    combined_data = np.concatenate((source_data, target_data), axis=0)
    combined_labels = np.concatenate((source_labels, target_labels), axis=0)
    combined_domain_labels = np.concatenate((source_domain_labels, target_domain_labels), axis=0)
    # combined_data = source_data
    # combined_labels = source_labels
    # combined_domain_labels = source_domain_labels

    # Shuffle combined data
    combined_data, combined_labels, combined_domain_labels = shuffle(combined_data, combined_labels, combined_domain_labels)
    
    # Training loop
    previous_domain_acc = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        domain_losses = 0
        enc_losses = 0
        class_losses = 0
        for step in range(0, len(combined_data), batch_size):
            #print("STEP", step, "/", len(combined_data))
            batch_data = combined_data[step:step+batch_size]
            batch_labels = combined_labels[step:step+batch_size]
            batch_domain_labels = combined_domain_labels[step:step+batch_size]

            ## new forward pass 
            with tf.GradientTape() as task_tape, tf.GradientTape() as enc_tape, tf.GradientTape() as disc_tape:
                # Forward pass
                features = model.feature_extractor(batch_data)
                label_pred = model.label_classifier(features)
                domain_pred = model.domain_classifier(features)

               # Compute label loss
                label_loss = tf.keras.losses.binary_crossentropy(batch_labels.reshape(-1,1), label_pred)

                # Compute domain loss
                domain_loss = tf.keras.losses.binary_crossentropy(batch_domain_labels.reshape(-1,1), domain_pred)
                
                label_loss = tf.reduce_mean(label_loss)
                domain_loss = tf.reduce_mean(domain_loss)
                enc_loss = label_loss - lambda_value * domain_loss

                class_losses += label_loss
                domain_losses += domain_loss
                enc_losses += enc_loss

            # Compute gradients
            trainable_vars_enc = model.feature_extractor.trainable_variables
            trainable_vars_task = model.label_classifier.trainable_variables
            trainable_vars_disc = model.domain_classifier.trainable_variables
            
            gradients_task = task_tape.gradient(label_loss, trainable_vars_task)
            gradients_enc = enc_tape.gradient(enc_loss, trainable_vars_enc)
            gradients_disc = disc_tape.gradient(domain_loss, trainable_vars_disc)
            
            # Update weights
            model.label_classifier.optimizer.apply_gradients(zip(gradients_task, trainable_vars_task))
            model.feature_extractor.optimizer.apply_gradients(zip(gradients_enc, trainable_vars_enc))
            model.domain_classifier.optimizer.apply_gradients(zip(gradients_disc, trainable_vars_disc))
        
            # Print training progress
            #print(f"Step {step+1}/{len(combined_data)}")
            #print(f"Step {step+1}/{len(combined_data)} - Label Loss: {label_loss}, Domain Loss: {domain_loss}")
        print(f"Label Loss: {class_losses}, Domain Loss: {domain_losses}, Encoder loss : {enc_losses}")

        # Compute accuracy on training set
        features = model.feature_extractor(combined_data)
        predicted_class_labels = model.label_classifier(features)
        predicted_domain_labels = model.domain_classifier(features)

        train_accuracy_class = np.mean((predicted_class_labels > 0.5) == combined_labels)  
        train_accuracy_domain = np.mean((predicted_domain_labels > 0.5) == combined_domain_labels)  
        print(f"Training Accuracy:  class - {train_accuracy_class} domain - {train_accuracy_domain}")
        #print("Shape class", predicted_class_labels.shape , "shape domain", predicted_domain_labels.shape)
        class_acc = accuracy_score(combined_labels, np.where(predicted_class_labels > 0.5, 1, 0))
        domain_acc = accuracy_score(combined_domain_labels, np.where(predicted_domain_labels > 0.5, 1, 0))
        print("Model acc class labels", class_acc)
        print("Model acc domain labels", domain_acc)

        if domain_acc > previous_domain_acc:
            ## Save the weigths 
            previous_domain_acc = domain_acc
            feature_extractor_weights_path = './feature_extractor_weights.weights.h5'  # Replace 'path_to_save_feature_extractor_weights.h5' with the desired path
            model.save_weights(feature_extractor_weights_path)

    print("Training finished.")

    
