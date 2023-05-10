from keras import utils
from keras import models
from keras import layers
from keras import activations
from .utils import load_imgnet_weights
from .layers import PatchEmbedding, AddCLSToken, AddPositionEmbedding, TransformerEncoder
from .config import BASIC_URL, VIT_B_CONFIG, VIT_L_CONFIG, WEIGHTS_CONFIG, PRE_TRAINED_WEIGHTS_FNAME


class ViT(models.Model):
    """Implementation of VisionTransformer Based on Keras
    Args:
        - image_size: Input Image Size, integer or tuple
        - patch_size: Size of each patch, integer or tuple
        - num_classes: Number of output classes
        - hidden_dim: The embedding dimension of each patch, it should be set to an integer multiple of the `atten_heads`
        - mlp_dim: The projection dimension of the mlp_block, it is generally 4 times that of `hidden_dim`
        - atten_heads: Number of self attention heads
        - encoder_depth: Number of the transformer encoder layer
        - dropout_rate: Dropout probability
        - activatiion: Activation function of mlp_head
        - pre_logits: Insert pre_Logits layer or not 
        - include_mlp_head: Insert mlp_head layer or not
    """
    def __init__(
        self, 
        image_size = 224,
        patch_size = 16,
        num_classes = 1000,
        hidden_dim = 768,
        mlp_dim = 3072,
        atten_heads = 12,
        encoder_depth = 12,
        dropout_rate = 0.,
        activation = "linear",
        pre_logits = True,
        include_mlp_head = True,
        **kwargs
        ):
        assert isinstance(image_size, int) or isinstance(image_size, tuple), "`image_size` should be int or tuple !"
        assert isinstance(patch_size, int) or isinstance(patch_size, tuple), "`patch_size` should be int or tuple !"
        assert hidden_dim%atten_heads==0, "hidden_dim and atten_heads do not match !"

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        if not kwargs.get("name"):
            config = {
                "hidden_dim": hidden_dim,
                "mlp_dim": mlp_dim,
                "atten_heads": atten_heads,
                "encoder_depth": encoder_depth,
                }
            size = "CUSTOM_SIZE"
            if config == VIT_B_CONFIG: size = "B"
            if config == VIT_L_CONFIG: size = "L" 
            patch = f"{patch_size[0]}" if patch_size[0]==patch_size[1] else f"{patch_size[0]}_{patch_size[1]}"
            image = f"{image_size[0]}" if image_size[0]==image_size[1] else f"{image_size[0]}_{image_size[1]}"
            kwargs["name"] = f"ViT-{size}-{patch}-{image}"

        super().__init__(**kwargs)
        self.image_size = image_size[:2]
        self.patch_size = patch_size[:2]
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.atten_heads = atten_heads
        self.encoder_depth = encoder_depth
        self.dropout_rate = dropout_rate
        self.activation = activations.get(activation)
        self.pre_logits = pre_logits
        self.include_mlp_head = include_mlp_head
        self.load_weights_inf = None
        self.build((None, *self.image_size, 3))

    def build(self, input_shape):
        self.patch_embedding=PatchEmbedding(self.patch_size, self.hidden_dim, name="patch_embedding")
        self.add_cls_token = AddCLSToken(self.hidden_dim, name="add_cls_token")
        self.position_embedding = AddPositionEmbedding(name="position_embedding")
        self.encoder_blocks = [
            TransformerEncoder(
                self.mlp_dim, 
                self.atten_heads, 
                self.dropout_rate,
                name=f"transformer_block_{i}"
                ) for i in range(self.encoder_depth)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.extract_token = layers.Lambda(lambda x: x[:,0], name="extract_token")
        if self.pre_logits: self.pre_logits = layers.Dense(self.hidden_dim, activation="tanh", name="pre_logits")
        if self.include_mlp_head: self.mlp_head = layers.Dense(self.num_classes, self.activation, name="mlp_head")
        super().build(input_shape)
        self.call(layers.Input(input_shape[1:]))

    def call(self, inputs):
        x = self.patch_embedding(inputs)
        x = self.add_cls_token(x)
        x = self.position_embedding(x)
        for encoder in self.encoder_blocks:
            x = encoder(x)
        x = self.layer_norm(x)
        x = self.extract_token(x)
        if self.pre_logits:
            x = self.pre_logits(x)
        if self.include_mlp_head:
            x = self.mlp_head(x)
        return x

    def load_weights(self, *kwargs):
        super().load_weights(*kwargs)
        self.load_weights_inf = {l.name:"loaded - h5 weights" for l in self.layers}

    def loading_summary(self):
        """ Print model information about loading imagenet pre training weights.
        ` - imagenet` means successful reading of imagenet pre training weights file
        ` - not pre trained` means that the model did not load the imagenet pre training weights file
        ` - mismatch` means that the pre training weights of the imagenet do not match the parameter shapes of the model layer
        ` - not found` means that there is no weights for the layer in the imagenet pre training weights file
        ` - h5 weights` means that the model is loaded with h5 weights
        """
        if self.load_weights_inf:
            inf = [(key+' '*(35-len(key))+value+"\n") for (key,value) in self.load_weights_inf.items()]
        else:
            inf = [(l.name+' '*(35-len(l.name))+"not loaded - not pre trained"+"\n") for l in self.layers]
        print(f'Model: "{self.name}"')
        print("-"*65)
        print("layers" + " "*29 + "load weights inf")
        print("="*65)
        print("\n".join(inf)[:-1])
        print("="*65)


def ViT_B16(
    image_size=None,
    num_classes=None,
    activation="linear",
    dropout_rate=0.1,
    pre_logits=None,
    include_mlp_head=True,
    pre_trained=True,
    weights="imagenet21k",
    **kwargs
    ):
    """Implementation of ViT_B16 Based on Keras.
    Args:
        - image_size: Input Image Size
        - num_classes: Number of output classes
        - activation: Activation function of mlp_head
        - dropout_rate: Dropout probability
        - pre_logits: Insert pre_Logits layer or not 
        - include_mlp_head: Insert mlp_head layer or not
        - pre_trained: using pre trained model weights or not
        the .npz weight file will be downloaded to "C:\\Users\\username\\.keras\\weights" when it is True
        - weights: This argument will determine the selection which dataset does the pre trained model come from
        this argument will set the values of argument `image_size`, `num_classes`, and `pre_logits` if they are None
        this argument should be set to one of "imagenet21k" and "imagenet21k+imagenet2012"
        "imagenet21k" >>> `image_size`=224, `num_classes`=21843, `pre_logits`=True
        "imagenet21k+imagenet2012" >>> `image_size`=384, `num_classes`=1000, `pre_logits`=False

    return:
        - ViT_B16: ViT with a patch_size of 16, hiddem_dim of 768, 
        mlp_dim of 3072, atten_heads of 12, and encoder layers of 12
    """
 
    if image_size==None: image_size=WEIGHTS_CONFIG.get(weights).get("IMAGE_SIZE")
    if num_classes==None: num_classes=WEIGHTS_CONFIG.get(weights).get("NUM_CLASSES")
    if pre_logits==None: pre_logits=WEIGHTS_CONFIG.get(weights).get("PRE_LOGITS")

    vit = ViT(
        **VIT_B_CONFIG,
        patch_size=16,
        image_size=image_size,
        num_classes=num_classes,
        activation=activation,
        dropout_rate=dropout_rate,
        pre_logits=pre_logits,
        include_mlp_head=include_mlp_head,
        **kwargs
        )

    if pre_trained:
        FNAME = PRE_TRAINED_WEIGHTS_FNAME["ViT-B16"][weights]
        weights = utils.get_file(FNAME, origin=BASIC_URL+FNAME, cache_subdir="weights")
        load_imgnet_weights(vit, weights)

    return vit


def ViT_B32(
    image_size=None,
    num_classes=None,
    activation="linear",
    dropout_rate=0.1,
    pre_logits=None,
    include_mlp_head=True,
    pre_trained=True,
    weights="imagenet21k",
    **kwargs
    ):
    """Implementation of ViT_B32 Based on Keras.
    Args:
        - image_size: Input Image Size
        - num_classes: Number of output classes
        - activation: Activation function of mlp_head
        - dropout_rate: Dropout probability
        - pre_logits: Insert pre_Logits layer or not 
        - include_mlp_head: Insert mlp_head layer or not
        - pre_trained: using pre trained model weights or not
        the .npz weight file will be downloaded to "C:\\Users\\username\\.keras\\weights" when it is True
        - weights: This argument will determine the selection which dataset does the pre trained model come from
        this argument will set the values of argument `image_size`, `num_classes`, and `pre_logits` if they are None
        this argument should be set to one of "imagenet21k" and "imagenet21k+imagenet2012"
        "imagenet21k" >>> `image_size`=224, `num_classes`=21843, `pre_logits`=True
        "imagenet21k+imagenet2012" >>> `image_size`=384, `num_classes`=1000, `pre_logits`=False

    return:
        - ViT_B32: ViT with a patch_size of 32, hiddem_dim of 768, 
        mlp_dim of 3072, atten_heads of 12, and encoder layers of 12
    """

    if image_size==None: image_size=WEIGHTS_CONFIG.get(weights).get("IMAGE_SIZE")
    if num_classes==None: num_classes=WEIGHTS_CONFIG.get(weights).get("NUM_CLASSES")
    if pre_logits==None: pre_logits=WEIGHTS_CONFIG.get(weights).get("PRE_LOGITS")

    vit = ViT(
        **VIT_B_CONFIG,
        patch_size=32,
        image_size=image_size,
        num_classes=num_classes,
        activation=activation,
        dropout_rate=dropout_rate,
        pre_logits=pre_logits,
        include_mlp_head=include_mlp_head,
        **kwargs,
        )

    if pre_trained:
        FNAME = PRE_TRAINED_WEIGHTS_FNAME["ViT-B32"][weights]
        weights = utils.get_file(FNAME, origin=BASIC_URL+FNAME, cache_subdir="weights")
        load_imgnet_weights(vit, weights)

    return vit


def ViT_L16(
    image_size=None,
    num_classes=None,
    activation="linear",
    dropout_rate=0.1,
    pre_logits=None,
    include_mlp_head=True,
    pre_trained=True,
    weights="imagenet21k",
    **kwargs
    ):
    """Implementation of ViT_L16 Based on Keras.
    Args:
        - image_size: Input Image Size
        - num_classes: Number of output classes
        - activation: Activation function of mlp_head
        - dropout_rate: Dropout probability
        - pre_logits: Insert pre_Logits layer or not 
        - include_mlp_head: Insert mlp_head layer or not
        - pre_trained: using pre trained model weights or not
        the .npz weight file will be downloaded to "C:\\Users\\username\\.keras\\weights" when it is True
        - weights: This argument will determine the selection which dataset does the pre trained model come from
        this argument will set the values of argument `image_size`, `num_classes`, and `pre_logits` if they are None
        this argument should be set to one of "imagenet21k" and "imagenet21k+imagenet2012"
        "imagenet21k" >>> `image_size`=224, `num_classes`=21843, `pre_logits`=True
        "imagenet21k+imagenet2012" >>> `image_size`=384, `num_classes`=1000, `pre_logits`=False

    return:
        - ViT_L16: ViT with a patch_size of 16, hiddem_dim of 1024, 
        mlp_dim of 4096, atten_heads of 16, and encoder layers of 24
    """

    if image_size==None: image_size=WEIGHTS_CONFIG.get(weights).get("IMAGE_SIZE")
    if num_classes==None: num_classes=WEIGHTS_CONFIG.get(weights).get("NUM_CLASSES")
    if pre_logits==None: pre_logits=WEIGHTS_CONFIG.get(weights).get("PRE_LOGITS")

    vit = ViT(
        **VIT_L_CONFIG,
        patch_size=16,
        image_size=image_size,
        num_classes=num_classes,
        activation=activation,
        dropout_rate=dropout_rate,
        pre_logits=pre_logits,
        include_mlp_head=include_mlp_head,
        **kwargs,
        )

    if pre_trained:
        FNAME = PRE_TRAINED_WEIGHTS_FNAME["ViT-L16"][weights]
        weights = utils.get_file(FNAME, origin=BASIC_URL+FNAME, cache_subdir="weights")
        load_imgnet_weights(vit, weights)

    return vit


def ViT_L32(
    image_size=None,
    num_classes=None,
    activation="linear",
    dropout_rate=0.1,
    pre_logits=None,
    include_mlp_head=True,
    pre_trained=True,
    weights="imagenet21k",
    **kwargs
    ):
    """Implementation of ViT_L32 Based on Keras.
    Args:
        - image_size: Input Image Size
        - num_classes: Number of output classes
        - activation: Activation function of mlp_head
        - dropout_rate: Dropout probability
        - pre_logits: Insert pre_Logits layer or not 
        - include_mlp_head: Insert mlp_head layer or not
        - pre_trained: using pre trained model weights or not
        the .npz weight file will be downloaded to "C:\\Users\\username\\.keras\\weights" when it is True
        - weights: This argument will determine the selection which dataset does the pre trained model come from
        this argument will set the values of argument `image_size`, `num_classes`, and `pre_logits` if they are None
        this argument should be set to one of "imagenet21k" and "imagenet21k+imagenet2012"
        "imagenet21k" >>> `image_size`=224, `num_classes`=21843, `pre_logits`=True
        "imagenet21k+imagenet2012" >>> `image_size`=384, `num_classes`=1000, `pre_logits`=False

    return:
        - ViT_L16: ViT with a patch_size of 32, hiddem_dim of 1024, 
        mlp_dim of 4096, atten_heads of 16, and encoder layers of 24
    """

    if image_size==None: image_size=WEIGHTS_CONFIG.get(weights).get("IMAGE_SIZE")
    if num_classes==None: num_classes=WEIGHTS_CONFIG.get(weights).get("NUM_CLASSES")
    if pre_logits==None: pre_logits=WEIGHTS_CONFIG.get(weights).get("PRE_LOGITS")

    vit = ViT(
        **VIT_L_CONFIG,
        patch_size=32,
        image_size=image_size,
        num_classes=num_classes,
        activation=activation,
        dropout_rate=dropout_rate,
        pre_logits=pre_logits,
        include_mlp_head=include_mlp_head,
        **kwargs,
        )

    if pre_trained:
        FNAME = PRE_TRAINED_WEIGHTS_FNAME["ViT-L32"][weights]
        weights = utils.get_file(FNAME, origin=BASIC_URL+FNAME, cache_subdir="weights")
        load_imgnet_weights(vit, weights)

    return vit