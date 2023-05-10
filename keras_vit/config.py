BASIC_URL = "https://github.com/djsaber/Keras-ViT/releases/download/v1.0.0/"

SIZE = ["ViT-B16", "ViT-B32", "ViT-L16", "ViT-L32"]

WEIGHTS = ["imagenet21k", "imagenet21k+imagenet2012"]

FNAMES = [
    "ViT-B_16_imagenet21k.npz", 
    "ViT-B_32_imagenet21k.npz",
    "ViT-L_16_imagenet21k.npz",
    "ViT-L_32_imagenet21k.npz",
    "ViT-B_16_imagenet21k+imagenet2012.npz",
    "ViT-B_32_imagenet21k+imagenet2012.npz",
    "ViT-L_16_imagenet21k+imagenet2012.npz",
    "ViT-L_32_imagenet21k+imagenet2012.npz"]

PRE_TRAINED_WEIGHTS_FNAME = {
    SIZE[0]: {WEIGHTS[0]: FNAMES[0], WEIGHTS[1]: FNAMES[4]},
    SIZE[1]: {WEIGHTS[0]: FNAMES[1], WEIGHTS[1]: FNAMES[5]},
    SIZE[2]: {WEIGHTS[0]: FNAMES[2], WEIGHTS[1]: FNAMES[6]},
    SIZE[3]: {WEIGHTS[0]: FNAMES[3], WEIGHTS[1]: FNAMES[7]},
    }

WEIGHTS_CONFIG = {
    WEIGHTS[0]: {"IMAGE_SIZE": 224, "NUM_CLASSES": 21843, "PRE_LOGITS": True},
    WEIGHTS[1]: {"IMAGE_SIZE": 384, "NUM_CLASSES": 1000, "PRE_LOGITS": False}
    }

VIT_B_CONFIG = {
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "atten_heads": 12,
    "encoder_depth": 12,
    }

VIT_L_CONFIG = {
    "hidden_dim": 1024,
    "mlp_dim": 4096,
    "atten_heads": 16,
    "encoder_depth": 24,
    }

FNAME_CLASSES_IN2012 = "imagenet2012.txt"