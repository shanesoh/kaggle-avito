from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import os
import numpy as np
from PIL import Image
import datetime
from itertools import izip_longest

def image_gen(img_array, size, batch_size):
    images = np.random.choice(img_array, batch_size)
    X_train = np.zeros((batch_size, size[0], size[1], size[2]))
    for idx, img in enumerate(images):
        try:
            processed = np.array(Image.open(img).resize(size[:2])).astype('float32')/255.
            if processed.ndim < 3:
                processed = np.repeat(processed[:, :, np.newaxis], 3, axis=2)
            X_train[idx] = processed
        except Exception as e:
            print e
            continue
    return X_train

image_dirs = ['./data/images/Images_1']
#image_dirs = ['./data/images/Images_1',
#              './data/images/Images_2',
#              './data/images/Images_3',
#              './data/images/Images_4',
#              './data/images/Images_5',
#              './data/images/Images_6',
#              './data/images/Images_7',
#              './data/images/Images_8',
#              './data/images/Images_9']
all_images = []
for image_dir in image_dirs:
    for fd in os.listdir(image_dir):
        for fn in os.listdir(os.path.join(image_dir, fd)):
            if os.path.splitext(fn)[1] == '.jpg':
                all_images.append(os.path.join(image_dir, fd, fn))

input_shape = (28, 28, 3)
batch_size = 100000
nb_epoch = 200

input_img = Input(shape=input_shape)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='tf')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(encoded)
x = UpSampling2D((2, 2), dim_ordering='tf')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
x = UpSampling2D((2, 2), dim_ordering='tf')(x)
x = Convolution2D(16, 3, 3, activation='relu', dim_ordering='tf')(x)
x = UpSampling2D((2, 2), dim_ordering='tf')(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', dim_ordering='tf')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#print("Training...")
#for n in range(nb_epoch):
#    train = image_gen(all_images, input_shape, batch_size)
#    loss = autoencoder.train_on_batch(train, train)
#    print("Epoch  %s: %s" % (n, loss))
#
#print("Saving model...")
#now = datetime.datetime.now()
#autoencoder.save_weights("autoencoder_%s.model" % now.strftime("%Y-%m-%d-%H-%M"))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

autoencoder.load_weights("autoencoder.model")
encoder = Model(input=input_img, output=encoded)

total = 0
for img_dir in image_dirs:
    print("Processing img_dir: %s" % img_dir)
    fn_array = []
    for img_subdir in os.listdir(img_dir):
        fn_array.extend(
            [os.path.join(img_dir, img_subdir, dir) for dir in os.listdir(os.path.join(img_dir, img_subdir))]
        )
    img_dir_idx = os.path.basename(img_dir).split('_')[1]
    fout = open('data/images/image_ae_%s.csv' % img_dir_idx, 'w')
    fout.write(','.join(['imageID', 'images_conv_ae']) + '\n')
    for images in chunker(fn_array, batch_size):
        X_test = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        try:
            for idx, img in enumerate(images):
                    processed = np.array(Image.open(img).resize(input_shape[:2])).astype('float32')/255.
                    if processed.ndim < 3:
                        processed = np.repeat(processed[:, :, np.newaxis], 3, axis=2)
                    X_test[idx] = processed
            output = encoder.predict(X_test)
            output = output.reshape([batch_size, 128])
            for idx in range(batch_size):
                img_idx = os.path.basename(images[idx]).split('.')[0]
                fout.write(','.join([img_idx, str(output[idx].tolist())]) + '\n')
        except Exception as e:
            print e
            continue
        total += batch_size
        print("Processed: %s" % total)

fout.close()
