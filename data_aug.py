from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os

datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        vertical_flip = False,
        fill_mode='nearest')

def data_aug(image_dir):
    train_csv = os.path.join(image_dir,"train.csv")
    f = open(train_csv,"r")
    lines = f.readlines()
    np.random.shuffle(lines)
    images = []
    labels = []
    labels_all = []
    label_id = []
    occlude_class = ["new_whale"]
    save_dir = imageName = os.path.join(image_dir,"train_aug_2")
    for i,line in enumerate(lines):
        #if(i>2):continue
        if(i==0): continue
        temp = line.strip().split(",")
        image_name = temp[0]
        imageName = os.path.join(image_dir,"train",image_name)
        if(not os.path.exists(imageName)): continue
        print(i,imageName)
        label=temp[1]
        if(label in occlude_class): continue
        img = load_img(imageName)  
        x = img_to_array(img) 
        x = x.reshape((1,) + x.shape)  
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_dir, save_prefix=image_name+"_"+label, save_format='jpg'):
            i += 1
            if i > 12:
                break  
    print("end")
    
data_aug("/nfs/project/whale/")