# coding=gbk

from PIL import Image
import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def load_data(path):
    with open(path, 'rb') as f:
        data_batch = pickle.load(f, encoding='bytes')  
        data = data_batch[bytes("data", encoding='utf-8')]
        labels = data_batch[bytes("labels", encoding='utf-8')]
        
        data = np.array(data, dtype=np.int8)
        labels = np.array(labels, dtype=np.int8)
        
        imgs_data = np.reshape(data, (10000,3,32,32))
        imgs = np.empty((10000,32,32,3), dtype=np.int8)
        imgs[:,:,:,0] = imgs_data[:,0,:,:]+128
        imgs[:,:,:,1] = imgs_data[:,1,:,:]+128
        imgs[:,:,:,2] = imgs_data[:,2,:,:]+128
        
        return imgs, labels


def load_cifar_10_label_name(path):
    with open(path+"/batches.meta", 'rb') as f:
        meta = pickle.load(f, encoding='bytes')  
        label_names = meta[bytes("label_names", encoding='utf-8')]
    return label_names


def load_cifar_10(path):
    train_batchs = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5"
        ]
    test_batchs = ["test_batch"]

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for batch in train_batchs:
        imgs, labels = load_data(path+"/"+batch)
        train_imgs.append(imgs)
        train_labels.append(labels)

    for batch in test_batchs:
        imgs, labels = load_data(path+"/"+batch)
        test_imgs.append(imgs)
        test_labels.append(labels)

    train_imgs = np.concatenate(train_imgs)
    train_labels = np.concatenate(train_labels)
    test_imgs = np.concatenate(test_imgs)
    test_labels = np.concatenate(test_labels)
    
    return (train_imgs, train_labels), (test_imgs, test_labels)


def preprocessing_function(resize=None):
    def apply(x):
        if resize:
            if len(x.shape)==3:
                x = Image.fromarray(np.uint8(x))
                x = x.resize(resize,Image.NEAREST)
                x = np.array(x, np.int8)
            elif len(x.shape)==4:
                new_x = np.empty((x.shape[0], *resize, 3), dtype=x.dtype)
                for i,s in enumerate(x):
                    new_s = Image.fromarray(np.uint8(s))
                    new_s = new_s.resize(resize,Image.NEAREST)
                    new_s = np.array(new_s, np.int8)
                    new_x[i] = new_s
                x = new_x
        x = (x+128)/127.5-1
        return x
    return apply


def my_data_gen(infinite_data_gen, resize=None):
    while True:
        imgs_batch, labels_batch = next(infinite_data_gen)
        imgs_batch = preprocessing_function(resize)(imgs_batch)
        yield imgs_batch, labels_batch


def cifar_10_data_gen(path, batch_size=32, data="train", resize=None, label_smooth=None):
    '''对cifar-10数据集构建的数据生成器
    '''
    (train_imgs, train_labels), (test_imgs, test_labels) = load_cifar_10(path)
    if data == "train":
        imgs, labels = train_imgs, to_categorical(train_labels, 10)
    if data == "test":
        imgs, labels = test_imgs, to_categorical(test_labels, 10)
    if label_smooth:
        labels = labels*(1-label_smooth) + (label_smooth/10)
 
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=0.2 if data=="train" else 0, 
        )
    data_gen.fit(imgs)

    if data == "train":
        train_data_gen = data_gen.flow(imgs, labels, batch_size, subset='training')
        valid_data_gen = data_gen.flow(imgs, labels, batch_size, subset='validation')
        my_train = my_data_gen(train_data_gen, resize)
        my_valid = my_data_gen(valid_data_gen, resize)
        return my_train, my_valid

    if data == "test":
        test_data_gen = data_gen.flow(imgs, labels, batch_size, shuffle=False)
        my_test = my_data_gen(test_data_gen, resize)
        return my_test