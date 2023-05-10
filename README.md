# keras-vit

这个包是基于Keras框架的Vision Transformer（ViT）实现。 ViT模型由论文 "[An image is worth 16x16 words: transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929.pdf)" 提出。这个包使用在imagenet21K数据集和imagenet21K+imagenet2012数据集上的预训练权重，它们是.npz格式的。

## **◈ 版本要求和安装**

- Python >= 3.7

- Keras >= 2.9

- ```
  pip install keras-vit
  ```

## **Q1: 能用这个包干什么？**

- 构建标准架构的预训练VisionTransformer（ViT）模型

- 构建自定义参数的ViT模型以适用于不同任务

## **Q2: 如何构建预训练 ViT模型？**

1. **快速构建预训练 ViTB16**
   
   ```
   from keras_vit.vit import ViT_B16
   vit = ViT_B16()
   ```
   
   > *预训练ViT有四种配置 ：ViT_B16，ViT_B32，ViT_L16 和 ViT_L32*
   > 
   > | 配置        | patch size | hiddem dim | mlp dim | attention heads | encoder depth |
   > |:---------:|:----------:|:----------:|:-------:|:---------------:|:-------------:|
   > | *ViT_B16* | 16×16      | 768        | 3072    | 12              | 12            |
   > | *ViT_B32* | 32×32      | 768        | 3072    | 12              | 12            |
   > | *ViT_L16* | 16×16      | 1024       | 4096    | 16              | 24            |
   > | *ViT_L32* | 32×32      | 1024       | 4096    | 16              | 24            |
   > 
   > *数据集 "imagenet21k" 和 "imagenet21k+imagenet2012" 的预训练权重对应的模型参数有些许不同，如下表所示：*
   > 
   > | weights                    | image size | classes | pre logits | known labels |
   > |:--------------------------:|:----------:|:-------:|:----------:|:------------:|
   > | *imagenet21k*              | 224        | 21843   | True       | False        |
   > | *imagenet21k+imagenet2012* | 384        | 1000    | False      | True         |

2. **构建不同数据集下的预训练ViTB16**
   
   ```
   from keras_vit.vit import ViT_B16
   vit_1 = ViT_B16(weights = "imagenet21k")
   vit_2 = ViT_B16(weights="imagenet21k+imagenet2012")
   ```
   
   > *预训练权重（.npz）文件会自动下载到：C:\Users\user_name\\.Keras\weights路径下。如果在下载过程意外中断，需要该路径下的文件删除并重新下载。*

3. **构建未进行预训练的ViT6**
   
   ```
   from keras_vit.vit import ViT_B16
   vit = ViT_B16(pre_trained=False)
   ```

4. **自定义参数构建预训练的ViT32**
   
   ```
   from keras_vit.vit import ViT_B32
   vit = ViT_B32(
       image_size = 128,
       num_classes = 12, 
       pre_logits = False,
       weights = "imagenet21k",
       )
   ```
   
   > *当改变了预训练模型的参数，模型中某些层的参数会发生改变，这些层就不再读取预训练权重，而是随机初始化。对于未发生改变的层，预训练权重参数会正常加载到这些层中。可以通过* `loading_summary()`*方法查看每一层的加载信息。*
   
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

## **Q3: 如何自定义构建ViT？**

1. **通过实例化 ViT 类来构建自定义ViT模型**
   
   ```
   from keras_vit.vit import ViT
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
   
   > *需要注意的是，*`hidden_dim`*参数需要能被* `atten_heads`*参数整除。*`image_size`*参数最好能被* `patch_size`*参数整除。*

2. **将预训练权重加载到自定义ViT模型中**
   
   ```
   from keras_vit import utils, vit
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

#### **Q4: 如何对预训练的ViT进行微调和直接用于图像分类 ？**

1. **微调**
   
   ```
   from keras_vit.vit import ViT_L16
   
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

2. **图像分类**
   
   ```
   from keras_vit import vit
   from keras_vit import utils
   
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
   
   > *需要注意的是，由于目前包中没有imagenet21k数据集的标签文件，因此在应用预先训练的ViT进行图像分类时，请设置* `“imagenet21-k+imagenet2012”`。
   > 
   > *若进行微调，则* `“imagenet21k”` *和* `“imagenet21k+imagenet2012”`*都可用。*

# 
