# ViT_Keras

This is a package that implements the ViT model based on Keras and Tensorflow. The ViT was proposed in the paper "[An image is worth 16x16 words: transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf)". This package uses pre trained weights on the imagenet21K and imagenet2012 datasets, which are in. npz format.

**Q1: What can you do with this package？**

- Build a pre trained standard specification ViT model.

- Customize and build any specification ViT model to suit your task.

**Q2: How to build a pre trained ViT？**

1. Quickly build a pre trained ViTB16
   
   ```
   from ViT_Keras.vit import ViT_B16
   vit = ViT_B16()
   ```
   
   > The pre trained ViT has 4 configurations: ViT_B16, ViT_B32, ViT_L16 and ViT_L32.
   > 
   > | config  | patch size | hiddem dim | mlp dim | attention heads | encoder depth |
   > |:-------:|:----------:|:----------:|:-------:|:---------------:|:-------------:|
   > | ViT_B16 | 16×16      | 768        | 3072    | 12              | 12            |
   > | ViT_B32 | 32×32      | 768        | 3072    | 12              | 12            |
   > | ViT_L16 | 16×16      | 1024       | 4096    | 16              | 24            |
   > | ViT_L32 | 32×32      | 1024       | 4096    | 16              | 24            |
   > 
   > The "imagenet21k" and "imagenet21k+imagenet2012" are slightly different, as shown in the table below.
   > 
   > | dataset                  | image size | classes | pre logits | known labels |
   > |:------------------------:|:----------:|:-------:|:----------:|:------------:|
   > | imagenet21k              | 224        | 21843   | True       | False        |
   > | imagenet21k+imagenet2012 | 384        | 1000    | False      | True         |

2. Build ViTB16 with differernt pre trained weights.
   
   ```
   from ViT_Keras.vit import ViT_B16
   vit_1 = ViT_B16(weights = "imagenet21k")
   vit_2 = ViT_B16(weights="imagenet21k+imagenet2012")
   ```

3. Build ViTB16 without pre trained weights
   
   ```
   from ViT_Keras.vit import ViT_B16
   vit = ViT_B16(pre_trained=False)
   ```
   
   > The pre training weights file will be downloaded to C:\Users\user_name\\. Keras\weights when "pre_trained = True".

4. Build pre trained ViTB32 with custom parameters
   
   ```
   from ViT_Keras.vit import ViT_B32
   vit = ViT_B32(
       image_size = 128,
       num_classes = 12, 
       pre_logits = False,
       weights = "imagenet21k",
       )
   ```
   
   > When you change some model parameters and some layers change, these layers will not load pre trained weights, the unchanged layers will still load pre trained weights. You can use loading_summary() to view specific information.
   
   ```
   vit.loading_summary()
   >>
   Model: "ViT-B-32-128"
   -----------------------------------------------------------------
   layers                             load weights inf
   =================================================================
   patch_embedding                    loaded
   
   add_cls_token                      loaded - imagenet
   
   position_embedding                 not loaded - mismatch
   
   transformer_block_0                loaded - imagenet
   
   transformer_block_1                loaded - imagenet
   
   transformer_block_2                loaded - imagenet
   
   transformer_block_3                loaded - imagenet
   
   transformer_block_4                loaded - imagenet
   
   transformer_block_5                loaded - imagenet
   
   transformer_block_6                loaded - imagenet
   
   transformer_block_7                loaded - imagenet
   
   transformer_block_8                loaded - imagenet
   
   transformer_block_9                loaded - imagenet
   
   transformer_block_10               loaded - imagenet
   
   transformer_block_11               loaded - imagenet
   
   layer_norm                         loaded - imagenet
   
   mlp_head                           not loaded - mismatch
   =================================================================
   ```

**Q3: How to build a custom ViT？**

1. Instantiating ViT classes to build custom ViT models
   
   ```
   from ViT_Keras.vit import ViT
   vit = ViT(
       image_size = 128,
       patch_size = 36,
       num_classes = 1,
       hidden_dim = 128,
       mlp_dim = 512,
       atten_heads = 32,
       encoder_depth = 4,
       dropout_rate = 0.1,
       activation = "sigmoid",
       pre_logits = True,
       include_mlp_head = True,
       )
   vit.summary()
   
   >>
   Model: "ViT-CUSTOM_SIZE-36-128"
   _________________________________________________________________
    Layer (type)                Output Shape              Param #
   =================================================================
    patch_embedding (PatchEmbed  (None, 9, 128)           497792
    ding)
   
    add_cls_token (AddCLSToken)  (None, 10, 128)          128
   
    position_embedding (AddPosi  (None, 10, 128)          1280
    tionEmbedding)
   
    transformer_block_0 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_1 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_2 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    transformer_block_3 (Transf  (None, 10, 128)          198272
    ormerEncoder)
   
    layer_norm (LayerNormalizat  (None, 10, 128)          256
    ion)
   
    extract_token (Lambda)      (None, 128)               0
   
    pre_logits (Dense)          (None, 128)               16512
   
    mlp_head (Dense)            (None, 1)                 129
   
   =================================================================
   Total params: 1,309,185
   Trainable params: 1,309,185
   Non-trainable params: 0
   _________________________________________________________________==========================
   ```
   
   > It should be noted that "hidden_dim" should be divisible by "atten_heads".
   > 
   > It is best to set "image_size" size that can be evenly divided by "patch_size".

2. Load pre trained weights for custom model
   
   ```
   from ViT_Keras import utils, vit
   vit_custom = vit.ViT(
       image_size=128,
       patch_size=8,
       encoder_depth=4
       )
   utils.load_imgnet_weights(vit_custom, "ViT-B_16_imagenet21k.npz")
   vit_custom.loading_summary()
   
   >>
   Model: "ViT-CUSTOM_SIZE-8-128"
   -----------------------------------------------------------------
   layers                             load weights inf
   =================================================================
   patch_embedding                    mismatch
   
   add_cls_token                      loaded - imagenet
   
   position_embedding                 not loaded - mismatch
   
   transformer_block_0                loaded - imagenet
   
   transformer_block_1                loaded - imagenet
   
   transformer_block_2                loaded - imagenet
   
   transformer_block_3                loaded - imagenet
   
   layer_norm                         loaded - imagenet
   
   pre_logits                         loaded - imagenet
   
   mlp_head                           not loaded - mismatch
   =================================================================
   ```

**Q4: Fine tuning or image classification on pre trained ViT ？**

1. Fine tuning pre trained ViT
   
   ```
   from ViT_Keras.vit import ViT_L16
   
   # Set parameters
   IMAGE_SIZE = ...
   NUM_CLASSES = ...
   ACTIVATION = ...
   ...
   
   # build ViT
   vit = ViT_B32(
       image_size = IMAGE_SIZE,
       num_classes = NUM_CLASSES, 
       activation = ACTIVATION,
       )
   
   # Compiling ViT
   vit.compile(
       optimizer = ...,
       loss = ...,
       metrics = ...
       )
   
   # Define train, valid and test data
   train_generator = ...
   valid_generator = ...
   test_generator  = ...
   
   # fine tuning ViT
   vit.fit(
       x = train_generator ,
       validation_data = valid_generator ,
       steps_per_epoch = ...,
       validation_steps = ...,
       )
   
   # testing
   vit.evaluate(x = test_generator, steps=...)
   ```

2. Applying  pre trained ViT for Image Classification
   
   ```
   from ViT_Keras import vit
   from ViT_Keras import utils
   
   # Get pre-trained vitb16
   vit_model = vit.ViT_B16(weights="imagenet21k+imagenet2012")
   
   # Load a picture
   img = utils.read_img("test.jpg", resize=vit_model.image_size)
   img = img.reshape((1,*vit_model.image_size,3))
   
   # Classifying
   y = vit_model.predict(img)
   classes = utils.get_imagenet2012_classes()
   print(classes[y[0].argmax()])
   ```
   
   > It should be noted that as there is currently no label for "imagenet21k", please use "imagenet21k+imagenet2012" when applying pre trained ViT. Both "imagenet21k" and "imagenet21k+imagenet2012" are available during the fine-tuning stage.

# 
