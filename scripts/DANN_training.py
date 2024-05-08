import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf 
import numpy as np 
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Conv1D, Dense, Flatten, InputLayer, AveragePooling1D, Reshape, BatchNormalization, Normalization, Dropout, Conv1DTranspose
from keras.layers import Input, Embedding, LeakyReLU, Conv2D, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras import models
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class GDANN(models.Model):
    def __init__(self, no_features, regularization_coeff=0.01, task = 'binary_classification'):
        super(GDANN, self).__init__()
        self.regularization_coeff = regularization_coeff
        self.task = task
        self.no_features = no_features
        input_shape = (no_features, 1)
        #input_shape = (no_features,)
        self.feature_extractor, feature_extractor_output = self.build_feature_extractor(input_shape)
        print(feature_extractor_output)
        self.label_classifier = self.build_label_classifier(feature_extractor_output)
        self.define_generator = self.build_generator(feature_extractor_output)
        self.domain_classifier = self.build_domain_discriminator(data_shape=feature_extractor_output)
    

    
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
        if self.task == 'binary_classification':
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
    
    def build_domain_discriminator(self, data_shape = (11,1), n_classes = 10):
        # generic features extracted representation always real 
        in_src_dist = Input(shape=data_shape)
        # original distribution of timestep 0 either fake or real 
        in_target_dist = Input(shape=data_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_dist, in_target_dist])
        disc = Dense(512, activation='relu')(merged)
        disc = Dense(1024, activation='relu')(disc)
        disc = Dropout(0.25)(disc)
        disc = Dense(512, activation='relu')(disc)
        disc = Dropout(0.25)(disc)
        disc = Dense(128, activation='relu')(disc)
        #real/fake output
        out1 = Dense(1, activation='sigmoid')(disc)
        #class label output 
        out2 = Dense(n_classes, activation='softmax')(disc)
        # define model
        model = Model([in_src_dist, in_target_dist], [out1, out2])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

        

    def build_generator(self, latent_dim, n_classes=10):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = self.no_features
        li = Dense(n_nodes, kernel_initializer=init)(li)
        # treat one datapoint as an additional feature map of size no_featurex1x1
        li = Reshape((self.no_features, 1, 1))(li)
        # image generator input
        in_datapoint = Input(shape=(latent_dim,))
        # foundation for a 11x1 datapoint, first no is number of feature maps
        n_nodes = 63 * self.no_features * 1
        gen = Dense(n_nodes, kernel_initializer=init)(in_datapoint)
        gen = Activation('relu')(gen)
        gen = Reshape((self.no_features, 1, 63))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        ## The output shape of merge should be 11x1x64 now upsample 11 and then downsample back to 11
        ## Here you need to do the encoding decoding from pix2pix
        gen = Conv1DTranspose(128, kernel_size=6, strides=2, activation='relu')(merge)
        gen = BatchNormalization()(gen)
        gen = Conv1DTranspose(256, kernel_size=8, strides=4, activation='relu')(gen)
        gen = BatchNormalization()(gen)
        gen = Conv1D(256, kernel_size = 10, strides=2, activation='relu')(gen)
        gen = Conv1D(128, kernel_size = 4, strides=2, activation='relu')(gen)
        gen = Flatten()(gen)
        gen = Dense(11)(gen)
        out_layer = Activation('sigmoid')(gen)
        # define model
        model = Model([in_datapoint, in_label], out_layer)
        return model
    
    def define_gan(g_model, d_model, data_shape):
        # make weights in the discriminator not trainable
        for layer in d_model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        # define the source image
        in_src = Input(shape=data_shape)
        # connect the source image to the generator input
        gen_out = g_model(in_src)
        # connect the source input and generator output to the discriminator input
        dis_out = d_model([in_src, gen_out])
        # define gan model as taking noise and label and outputting real/fake and label outputs
        model = Model(in_src, [dis_out, gen_out])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy','sparse_categorical_crossentropy','mae'], optimizer=opt)
        return model


def train_dann_model(model, source_data, source_labels, target_data, target_labels, num_epochs=10, batch_size=512, learning_rate=1e-5, lambda_value=0.1, data_generator = "Perdomo"):
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
            feature_extractor_weights_path = f'./feature_extractor_weights_{data_generator}.weights.h5' 
            model.save_weights(feature_extractor_weights_path)

    print("Training finished.")


# select real samples
def generate_real_samples(dataset, map_of_idx_to_og, labels, n_samples):
    # Generate random indices
    random_indices = np.random.sample(range(len(dataset)), n_samples)
    datapoints = [dataset[idx] for idx in random_indices]
    # Select corresponding points from the original distribution in t0
    original_datapoints = []
    for idx in random_indices:
        original_dataset_index = map_of_idx_to_og[idx]
        original_datapoints.extend(dataset[original_dataset_index[0]][original_dataset_index[1]])
   
    return datapoints, original_datapoints 
    
def train_architecture(model, data, labels, num_epochs = 10, batch_size = 512):
    # Concatenate all inner lists into one big list for data and labels
    concatenated_data = [item for sublist in data for item in sublist]
    concatenated_labels = [item for sublist in labels for item in sublist] 
    
    # Create a corresponding list of labels
    concatenated_domain_labels = []
    # Create a mapping to track original instance indices
    original_instance_mapping = []

    for i, sublist in enumerate(data):
        original_instance_mapping.extend([(i, j) for j in range(len(sublist))])
        concatenated_domain_labels.extend([i] * len(sublist))

    # combined_data, combined_class_labels, combined_domain_labels = shuffle(concatenated_data, concatenated_labels, concatenated_domain_labels)
    # No shuffling for now as you need to be able to map one point from dist t=0 to its version in t1, t2 and so on 
    combined_data, combined_class_labels, combined_domain_labels = shuffle(concatenated_data, concatenated_labels, concatenated_domain_labels)
    
    #calculate the number of batches per training epoch
    bat_per_epo = int(combined_data.shape[0] / batch_size)
	# calculate the number of training iterations
    n_steps = bat_per_epo * num_epochs
	# calculate the size of half a batch of samples
    half_batch = int(batch_size / 2)
	# manually enumerate epochs
    for i in range(n_steps):
		# get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
        _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
		# generate 'fake' examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
        _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		# prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
        _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        