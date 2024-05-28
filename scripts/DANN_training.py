import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf 
import numpy as np 
import pandas as pd 
import sys
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers import Conv1D, Dense, Flatten, InputLayer, AveragePooling1D, Reshape, BatchNormalization, Normalization, Dropout, Conv1DTranspose
from keras.layers import Input, Embedding, LeakyReLU, Conv2D, Conv2DTranspose, Activation, Concatenate
from tensorflow.keras import models
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from keras.utils import plot_model


class GDANN(models.Model):
    def __init__(self, no_features, no_domain_classes = 5, task = 'binary_classification'):
        super(GDANN, self).__init__()
        self.task = task
        self.no_features = no_features
        input_shape = (no_features, 1)
        #input_shape = (no_features,)
        self.feature_extractor, feature_extractor_output = self.build_feature_extractor(input_shape)
        # print(feature_extractor_output)
        self.label_classifier = self.build_label_classifier(feature_extractor_output)
        self.discriminator = self.build_domain_discriminator(data_shape=(feature_extractor_output,1), n_classes=no_domain_classes)
        self.generator = self.build_generator(input_shape=(feature_extractor_output,1), n_classes= no_domain_classes)
        self.gan_model = self.build_gan(self.generator, self.discriminator, data_shape = (feature_extractor_output,1))
        

       
    
    def build_feature_extractor(self, input_shape):
        model = keras.models.Sequential()
        #model.add(InputLayer(shape = input_shape))
        model.add(Input(input_shape))
        model.add(Conv1DTranspose(32, kernel_size=6, strides=2, input_shape = input_shape, activation='relu'))# kernel_regularizer=l2(self.regularization_coeff)
        model.add(BatchNormalization())
        model.add(Conv1DTranspose(64, kernel_size=8, strides=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv1D(32, kernel_size=6, strides=14, activation='relu'))
        model.add(Flatten())
        model.add(Reshape((256, 1)))
        model.add(AveragePooling1D(pool_size=37, strides=20))
        #model.add(MaxPooling1D(pool_size=37, strides=20))
        model.add(Flatten())
        model.add(Dense(11, input_shape = input_shape))
        model.add(Normalization(mean = np.zeros(11), variance= np.ones(11)))
        #model.summary()
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer)
        ### Tune pool_size and strides according to output = (len(flatten) + 1 - pool_size) / stride
        output_shape = model.output_shape[1]
        plot_model(model, to_file='feature_extractor.png', show_shapes=True, show_layer_names=True)
        return model, output_shape
    
    def build_label_classifier(self, input_shape):
        if self.task == 'binary_classification':
            model = tf.keras.Sequential()
            #model.add(InputLayer(shape = (input_shape,)))
            model.add(Input((input_shape,)))
            model.add(Dense(256,activation= 'relu')) # kernel_regularizer=l2(self.regularization_coeff)
            model.add(Dense(512,activation='relu')) # , kernel_regularizer=l2(self.regularization_coeff)
            model.add(Dense(64,activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            optimizer = tf.keras.optimizers.Adam()
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
            #model.summary()
        return model
    
    def build_domain_discriminator(self, data_shape = (11,1), n_classes = 10):
        # # generic features extracted representation always real 
        # in_src_dist = Input(shape=data_shape)
        # # original distribution of timestep 0 either fake or real 
        # in_target_dist = Input(shape=data_shape)
        # # concatenate images channel-wise
        # merged = Concatenate()([in_src_dist, in_target_dist])
        # disc = Dense(512, activation='relu')(merged)
        # disc = Dense(1024, activation='relu')(disc)
        # disc = Dropout(0.25)(disc)
        # disc = Dense(512, activation='relu')(disc)
        # disc = Dropout(0.25)(disc)
        # disc = Dense(128, activation='relu')(disc)
        # #real/fake output
        # disc = Flatten()(disc)
        # out1 = Dense(1, activation='sigmoid', name = 'real_fake')(disc)
        # #class label output 
        # out2 = Dense(n_classes, activation='softmax', name = 'category')(disc)
        # # define model
        # model = Model([in_src_dist, in_target_dist], [out1, out2])
        # # compile model
        # opt = Adam(learning_rate=0.0002, beta_1=0.5)
        # #model.summary()
        # model.compile(
        #     loss ={'real_fake':'binary_crossentropy', 'category':'sparse_categorical_crossentropy'},
        #     optimizer=opt, metrics = ['accuracy', 'accuracy'])
        
        # plot_model(model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)




        ### SAME CODE AS ABOVE BUT FEWER PARAMETERS 
        # # generic features extracted representation always real 
        # in_src_dist = Input(shape=data_shape)
        # # original distribution of timestep 0 either fake or real 
        # in_target_dist = Input(shape=data_shape)
        # # concatenate images channel-wise
        # merged = Concatenate()([in_src_dist, in_target_dist])
        # disc = Dense(64, activation='leaky_relu')(merged)
        # disc = BatchNormalization()(disc)
        # disc = Dropout(0.25)(disc)
        # disc = Dense(128, activation='leaky_relu')(disc)
        # #disc = BatchNormalization()(disc)
        # disc = Dropout(0.25)(disc)
        # disc = Dense(512, activation='leaky_relu')(disc)
        # #disc = BatchNormalization()(disc)
        # disc = Dropout(0.25)(disc)
        # disc = Dense(512, activation='leaky_relu')(disc)
        # #disc = BatchNormalization()(disc)
        # #real/fake output
        # disc = Flatten()(disc)
        # out1 = Dense(1, activation='sigmoid', name = 'real_fake')(disc)
        # #class label output 
        # out2 = Dense(n_classes, activation='softmax', name = 'category')(disc)
        # # define model
        # model = Model([in_src_dist, in_target_dist], [out1, out2])
        # # compile model
        # opt = Adam(learning_rate=0.00001)
        # #model.summary()
        # model.compile(
        #     loss ={'real_fake':'binary_crossentropy', 'category':'sparse_categorical_crossentropy'},
        #     optimizer=opt, metrics = ['accuracy', 'accuracy'])
        
        # plot_model(model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)



        ## Model where og distriubtion and feature map are not concatenated
        # generic features extracted representation always real 
        in_src_dist = Input(shape=data_shape, name='timestamp_0')
        # original distribution of timestep 0 either fake or real 
        in_target_dist = Input(shape=data_shape, name = 'features')
        # concatenate images channel-wise
        bin = Dense(64, activation='leaky_relu')(in_src_dist)
        bin = BatchNormalization()(bin)
        bin = Dropout(0.25)(bin)
        bin = Dense(128, activation='leaky_relu')(bin)
        #disc = BatchNormalization()(disc)
        bin = Dropout(0.25)(bin)
        bin = Dense(512, activation='leaky_relu')(bin)
        #disc = BatchNormalization()(disc)
        bin = Dropout(0.25)(bin)
        bin = Dense(512, activation='leaky_relu')(bin)
        #disc = BatchNormalization()(disc)
        #real/fake output
        bin = Flatten()(bin)
        out1 = Dense(1, activation='sigmoid', name = 'real_fake')(bin)

        multi_class = Dense(64, activation='leaky_relu')(in_target_dist)
        multi_class = BatchNormalization()(multi_class)
        multi_class = Dropout(0.25)(multi_class)
        multi_class = Dense(128, activation='leaky_relu')(multi_class)
        #disc = BatchNormalization()(disc)
        multi_class = Dropout(0.25)(multi_class)
        multi_class = Dense(512, activation='leaky_relu')(multi_class)
        #disc = BatchNormalization()(disc)
        multi_class = Dropout(0.25)(multi_class)
        multi_class = Dense(512, activation='leaky_relu')(multi_class)
        #disc = BatchNormalization()(disc)
        #real/fake output
        multi_class = Flatten()(multi_class)

        #class label output 
        #out2 = Dense(n_classes, activation='softmax', name = 'category')(multi_class)
        out2 = Dense(n_classes, activation='sigmoid', name = 'category')(multi_class)
        # define model
        model = Model([in_src_dist, in_target_dist], [out1, out2])
        # compile model
        opt = Adam(learning_rate=0.0001)
        #model.summary()
        model.compile(
            loss ={'real_fake':'binary_crossentropy', 'category':'sparse_categorical_crossentropy'},
            optimizer=opt, metrics = ['accuracy', 'accuracy'])
        
        plot_model(model, to_file='discriminator.png', show_shapes=True, show_layer_names=True)
        return model

    def build_generator(self, input_shape, n_classes=10):
        # # weight initialization
        # init = RandomNormal(stddev=0.02)
        # # label input
        # in_label = Input(shape=(1,))
        # # embedding for categorical input
        # li = Embedding(n_classes, 50)(in_label)
        # # linear multiplication
        # n_nodes = self.no_features
        # li = Dense(n_nodes, kernel_initializer=init)(li)
        # # treat one datapoint as an additional feature map of size no_featurex1x1
        # li = Reshape((self.no_features,1))(li)
        # # image generator input
        # in_datapoint = Input(shape=input_shape)
        # # foundation for a 11x1 datapoint, first no is number of feature maps
        # n_nodes = 64 
        # gen = Dense(n_nodes, kernel_initializer=init)(in_datapoint)
        # #gen = Activation('relu')(gen)
        # gen = LeakyReLU()(gen)
        # gen = Flatten()(gen)
        # gen = Reshape((self.no_features,64))(gen)
        # # merge image gen and label input
        # merge = Concatenate()([gen, li])
        # ## The output shape of merge should be 11x1x64 now upsample 11 and then downsample back to 11
        # ## Here you need to do the encoding decoding from pix2pix
        # gen = Conv1DTranspose(128, kernel_size=6, strides=2)(merge)
        # gen = LeakyReLU()(gen)
        # gen = BatchNormalization()(gen)
        # gen = Conv1DTranspose(256, kernel_size=8, strides=4)(gen)
        # gen = LeakyReLU()(gen)
        # gen = BatchNormalization()(gen)
        # gen = Conv1D(256, kernel_size = 10, strides=2)(gen)
        # gen = LeakyReLU()(gen)
        # gen = BatchNormalization()(gen)
        # gen = Conv1D(128, kernel_size = 4, strides=2)(gen)
        # gen = LeakyReLU()(gen)
        # gen = BatchNormalization()(gen)
        # gen = Flatten()(gen)
        # gen = Dense(11)(gen)
        # out_layer = Activation('sigmoid')(gen)
        # # define model
        # model = Model([in_datapoint, in_label], out_layer)



        #### SAME CODE AS ABOVE BUT FEWER PARAMETERS  

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
        li = Reshape((self.no_features,1))(li)
        # image generator input
        in_datapoint = Input(shape=input_shape)
        # foundation for a 11x1 datapoint, first no is number of feature maps
        n_nodes = 64
        gen = Dense(n_nodes, kernel_initializer=init)(in_datapoint)
        #gen = Activation('relu')(gen)
        gen = LeakyReLU()(gen)
        gen = Flatten()(gen)
        gen = Reshape((self.no_features,64))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        ## The output shape of merge should be 11x1x64 now upsample 11 and then downsample back to 11
        ## Here you need to do the encoding decoding from pix2pix
        gen = Conv1DTranspose(128, kernel_size=6, strides=2)(merge)
        gen = LeakyReLU()(gen)
        gen = BatchNormalization()(gen)
        gen = Conv1DTranspose(256, kernel_size=8, strides=4)(gen)
        gen = LeakyReLU()(gen)
        gen = BatchNormalization()(gen)
        gen = Conv1D(256, kernel_size = 10, strides=2)(gen)
        gen = LeakyReLU()(gen)
        gen = BatchNormalization()(gen)
        gen = Conv1D(128, kernel_size = 4, strides=2)(gen)
        gen = LeakyReLU()(gen)
        gen = BatchNormalization()(gen)
        gen = Flatten()(gen)
        gen = Dense(11)(gen)
        #test different activation
        #Either map it to (-3,3)
        #scaling_factor = 10
        #out_layer = Activation(lambda x: scaling_factor * tf.keras.activations.tanh(x))(gen)
        #here it gets mapped to (-1,1)
        #out_layer = Activation('tanh')(gen)
        # define model
        out_layer = Activation('linear')(gen)
        model = Model([in_datapoint, in_label], out_layer)


        return model
    
    def build_gan(self, g_model, d_model, data_shape):
               
        # define the source image
        in_src = Input(shape=data_shape)
        #define class 
        in_class = Input(shape = (1,))
        # connect the source image to the generator input
        gen_out = g_model([in_src, in_class])
        # connect the source input and generator output to the discriminator input
        dis_output = d_model([gen_out, in_src])
        # define gan model as taking noise and label and outputting real/fake and label outputs
        model = Model(inputs = [in_src ,in_class], outputs = [dis_output[0], dis_output[1], gen_out])
        # compile model
        opt = Adam(learning_rate=0.0004)
        model.compile(loss=['binary_crossentropy','sparse_categorical_crossentropy','mae'], optimizer=opt)
        #plot_model(model, to_file='GAN.png', show_shapes=True, show_layer_names=True)
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
def generate_real_samples(dataset, first_dist, len_single_domain, c_labels, d_labels, n_samples):
    # Generate random indices
    random_indices = np.random.choice(range(len(dataset)), n_samples)
    datapoints = np.array([dataset[idx] for idx in random_indices])
    #Find corresponding indices in the original dataset
    og_indices = random_indices % len_single_domain
    original_datapoints = first_dist[og_indices]
    # Same for class and domain labels 
    class_labels = np.array([c_labels[idx] for idx in random_indices])
    domain_labels = np.array([d_labels[idx] for idx in random_indices])

    return datapoints, original_datapoints, class_labels, domain_labels

    
def train_architecture(model,first_dist, data, labels,lam = 100, lambda_2 = 1,  num_epochs = 100, batch_size = 1024):
    domain_labels = []
    iterations_data = []

    len_single_domain = len(data[0])
    for i, datapoints in enumerate(data):
        domain_labels.extend([i] * len(datapoints))
    
    combined_domain_labels = np.array(domain_labels)   
    combined_data = np.concatenate(data)
    combined_class_labels = np.concatenate(labels) 
    
    #calculate the number of batches per training epoch
    bat_per_epo = int(len(combined_data) / batch_size)
	# calculate the number of training iterations
    n_steps = bat_per_epo * num_epochs
	# calculate the size of half a batch of samples
    half_batch = int(batch_size / 2)
	# manually enumerate epochs
    for i in range(n_steps):
        print(f"Step no {i}/{n_steps}")
        # get randomly selected 'real' samples
        datapoints, original_points,class_labels, domain_labels  = generate_real_samples(combined_data,first_dist, len_single_domain, combined_class_labels, combined_domain_labels, half_batch)
        #datapoints, original_points, class_labels, domain_labels = shuffle(datapoints, original_points, class_labels, domain_labels)

        with tf.GradientTape() as feature_extractor_tape, tf.GradientTape() as label_classifier_tape: 
            # Pass forward through the discriminator
            with tf.GradientTape() as discriminator_tape: 
                for layer in model.discriminator.layers:
                    if not isinstance(layer, BatchNormalization):
                        layer.trainable = True
                    # if layer.name == "dense_8":
                    #     print("\n")
                    #     print(layer.name)
                    #     print(layer.get_weights())

                domain_labels_tensor = tf.convert_to_tensor(domain_labels)

                
                #Train discriminator on real distributions
                features = model.feature_extractor(datapoints)
                discriminator_output = model.discriminator([tf.convert_to_tensor(original_points), features])
                binary_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(discriminator_output[0]), discriminator_output[0])
                #category_loss_real = tf.keras.losses.sparse_categorical_crossentropy(domain_labels_tensor, discriminator_output[1])
                category_loss_real = tf.keras.losses.binary_crossentropy(tf.reshape(domain_labels_tensor, (-1,1)), discriminator_output[1])
                
                #Train discriminator on generated distributions
                generated_t0 = model.generator([features, tf.convert_to_tensor(domain_labels)])
                discriminator_output_fake = model.discriminator([generated_t0, features])
                binary_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(discriminator_output_fake[0]), discriminator_output_fake[0])
                #category_loss_fake = tf.keras.losses.sparse_categorical_crossentropy(domain_labels_tensor, discriminator_output_fake[1])
                category_loss_fake = tf.keras.losses.binary_crossentropy(tf.reshape(domain_labels_tensor, (-1,1)), discriminator_output_fake[1])

                discriminators_loss = tf.reduce_mean(binary_loss_real + category_loss_real + binary_loss_fake + category_loss_fake)
            
            
            
            discriminators_trainable_vars = model.discriminator.trainable_variables
            grads = discriminator_tape.gradient(discriminators_loss, discriminators_trainable_vars)

            #print("DISCRIMINATOR GRADIENTS", grads)

            model.discriminator.optimizer.apply_gradients(zip(grads, discriminators_trainable_vars))
            
            #Pass forward through the generator
            with tf.GradientTape() as generator_tape: 

                for layer in model.discriminator.layers:
                    if not isinstance(layer, BatchNormalization):
                        layer.trainable = False


                generated_t0 = model.generator([features, tf.convert_to_tensor(domain_labels)])
                discriminator_output = model.discriminator([generated_t0, features])
                generator_binary_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(discriminator_output[0]), discriminator_output[0])
                #generator_category_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.convert_to_tensor(domain_labels),discriminator_output[1])
                generator_category_loss = tf.keras.losses.binary_crossentropy(tf.reshape(tf.convert_to_tensor(domain_labels),(-1,1)),discriminator_output[1])
                #mae_between_distributions = np.mean(np.abs(original_points - generated_t0), axis = 1)
                mae_between_distributions = tf.reduce_mean(tf.abs(original_points - generated_t0), axis = 1)
                #generator_loss = tf.reduce_mean(generator_binary_loss + generator_category_loss) + lam * mae_between_distributions
                generator_loss = tf.reduce_mean(generator_binary_loss + generator_category_loss) + lam * mae_between_distributions
                
                if i > 350:       
                    generator_trainable_vars = model.gan_model.trainable_variables
                    grads = generator_tape.gradient(generator_loss, generator_trainable_vars)
                    model.gan_model.optimizer.apply_gradients(zip(grads, generator_trainable_vars))

            label_pred = model.label_classifier(features)
            # print("Label_pred", label_pred[:10])
            # predicted_class_labels = tf.clip_by_value(label_pred, 0.0, 1.0)
            # print("Label_pred", predicted_class_labels[:10])
            label_loss = tf.keras.losses.binary_crossentropy(class_labels.reshape(-1,1), label_pred)
            encoder_loss = tf.reduce_mean(label_loss - lambda_2 * (category_loss_real + category_loss_fake))
         
        # Compute gradients
        if i <= 350:
            trainable_vars_enc = model.feature_extractor.trainable_variables
            trainable_vars_task = model.label_classifier.trainable_variables
            
            gradients_label_classifier = label_classifier_tape.gradient(label_loss, trainable_vars_task)
            gradients_feature_extractor = feature_extractor_tape.gradient(encoder_loss, trainable_vars_enc)

            
            # Update weights
            model.label_classifier.optimizer.apply_gradients(zip(gradients_label_classifier, trainable_vars_task))
            model.feature_extractor.optimizer.apply_gradients(zip(gradients_feature_extractor, trainable_vars_enc))
       


        #Compute the accuracies on the training set 
        # Compute accuracy on training set
        if i % 25 == 0 and i !=0 :
            if i == 25:
                num_points_20_percent = int(0.2 * len(combined_data))
                random_indices = np.random.choice(len(combined_data), num_points_20_percent, replace=False)

            test_data = combined_data[random_indices]
            test_domain_labels = combined_domain_labels[random_indices]
            test_class_labels = combined_class_labels[random_indices]
            features = model.feature_extractor(test_data)

            predicted_class_labels_probabilities = model.label_classifier(features)
            predicted_class_labels = tf.cast(predicted_class_labels_probabilities >= 0.5, tf.int32)
            train_accuracy_class = accuracy_score(test_class_labels, predicted_class_labels)

            generated_t0 = model.generator([features, tf.convert_to_tensor(test_domain_labels)])
            disc_output = model.discriminator([generated_t0, features])
            predicted_domain_probabilities = disc_output[1]
            #predicted_domain_class = np.argmax(predicted_domain_probabilities, axis=1)
            predicted_domain_class = tf.cast(predicted_domain_probabilities >= 0.5, tf.int32)

            #print("Points", test_data[:10])
            # print("Generated points", generated_t0[:5])
            # print("Original X points", first_dist[random_indices % int(0.2 * len(combined_data))])

            print("Category labels probs", predicted_domain_probabilities[:10])
            print("Category labels", predicted_domain_class[:10])
            print("True labels", test_domain_labels[:10])

            train_accuracy_domain = accuracy_score(predicted_domain_class, test_domain_labels)
            
            print("Probs", predicted_class_labels_probabilities[:10])
            print("Labels", predicted_class_labels[:10])
            print("True labels ", test_class_labels[:10])

            print("\n")
            print(f"Training Accuracy:  class - {train_accuracy_class}")
            print(f"Domain classification accuracy  - {train_accuracy_domain}")
            print("\n")
            #PRINT THE LOSSES 

            print("Discriminator loss", tf.reduce_mean(discriminators_loss).numpy())
            print("Generator loss", tf.reduce_mean(generator_loss).numpy())
            print("Generator loss from classification", tf.reduce_mean(generator_binary_loss).numpy())
            print("Generator loss from distance", tf.reduce_mean(lam * mae_between_distributions).numpy() )
            print("Feature extractor loss", tf.reduce_mean(encoder_loss).numpy())
            print("Label loss", tf.reduce_mean(label_loss).numpy())
            
            # Sample data for multiple iterations
            

            # Sample data for one iteration
            iteration_data = {
                "Probabilities": predicted_domain_probabilities,
                "Predicted labels": predicted_domain_class,
                "Domain_Classification_Accuracy": train_accuracy_domain,
                "Step number": i,
                "Generator mae loss": tf.reduce_mean(generator_loss).numpy(),
                "Generator real/fake loss": tf.reduce_mean(generator_binary_loss).numpy(),
                "Generator category loss": tf.reduce_mean(generator_category_loss).numpy(),
                "Discriminator loss": tf.reduce_mean(discriminators_loss).numpy(),
                "Label loss": tf.reduce_mean(label_loss).numpy(),
                "Class accuracy": train_accuracy_class,
                "Disc real loss": tf.reduce_mean(binary_loss_real).numpy(),
                "Disc fake loss": tf.reduce_mean(binary_loss_fake).numpy(),
                "Disc category loss real": tf.reduce_mean(category_loss_real).numpy(),
                "Disc category loss fake": tf.reduce_mean(category_loss_fake).numpy()
            }

            iterations_data.append(iteration_data)

    # Convert to DataFrame
    dfs = []
    for i, iteration in enumerate(iterations_data):
        print("Iteration Probabilities", iteration["Probabilities"])
        df = pd.DataFrame(iteration["Probabilities"], columns=[f"Probability_{j+1}" for j in range(len(iteration["Probabilities"][0]))])
        df['Predicted labels'] = iteration['Predicted labels']
        df['Domain_Classification_Accuracy'] = iteration["Domain_Classification_Accuracy"]
        df['Step number'] = iteration["Step number"]
        df['Generator mae loss'] = iteration["Generator mae loss"]
        df['Generator real/fake loss'] = iteration['Generator real/fake loss']
        df['Generator class loss'] = iteration['Generator category loss']
        df['Dicriminator loss'] = iteration["Discriminator loss"]
        df['Label loss'] = iteration['Label loss']
        df['Class accuracy'] = iteration['Class accuracy']
        df['Disc real loss'] = iteration['Disc real loss']
        df['Disc fake loss'] = iteration ['Disc fake loss']
        df['Disc category loss real'] = iteration['Disc category loss real']
        df['Disc category loss fake'] = iteration['Disc category loss fake']
        dfs.append(df)

    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_csv('concatenated_dataframe.csv', index=False)

    save_path = f'./GDANN_arch.weights.h5' 
    model.save_weights(save_path)
    print("Training finished.")

        