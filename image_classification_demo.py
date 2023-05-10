from keras_vit import vit
from keras_vit import utils


# get pre-trained vitb16
vit_model = vit.ViT_B16(weights="imagenet21k+imagenet2012")

# Load a picture
img = utils.read_img("test.jpg", resize=vit_model.image_size)
img = img.reshape((1,*vit_model.image_size,3))

# predict
y = vit_model.predict(img)
classes = utils.get_imagenet2012_classes()

# print class of the picture
print(classes[y[0].argmax()])
