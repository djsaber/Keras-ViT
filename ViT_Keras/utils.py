import cv2
from keras import utils
from keras import backend as K
from .config import BASIC_URL, FNAME_CLASSES_IN2012


def load_imgnet_weights(keras_model, weights_npz):
    """Load the imagenet pre training weights (.npz) for ViT Keras model.
    Weights file, for example: "ViT-B_16_imagenet21k.npz", "ViT-L_32_imagenet21k+imagenet2012.npz"
    More weights files: https://github.com/djsaber/Keras-ViT/releases/tag/v1.0.0
    args:
        - keras_model: The keras ViT models that need to load pre trained weights
        - weights_npz: The .npz weights file
    return:
        - load_inf: Dict of information about loading weights for each layer of the keras_model
    """
    params_dict = K.np.load(weights_npz, allow_pickle=False)
    keys = set(params_dict.keys())
    transformer_blocks_num = len([k for k in keys if k.startswith("Transformer/encoderblock_")])//16
    
    patch_emb_weights = [
        params_dict["embedding/kernel"],
        params_dict["embedding/bias"]]
    cls_token_weights = [params_dict["cls"]]
    position_emb_weights = [params_dict["Transformer/posembed_input/pos_embedding"]]
    transformer_block_weights = [
        [params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/query/bias"],
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/key/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/value/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MultiHeadDotProductAttention_1/out/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_0/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/kernel"], 
         params_dict[f"Transformer/encoderblock_{i}/MlpBlock_3/Dense_1/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/LayerNorm_0/scale"], 
         params_dict[f"Transformer/encoderblock_{i}/LayerNorm_0/bias"], 
         params_dict[f"Transformer/encoderblock_{i}/LayerNorm_2/scale"], 
         params_dict[f"Transformer/encoderblock_{i}/LayerNorm_2/bias"]
         ]  for i in range(transformer_blocks_num)]
    layer_norm_weights = [
        params_dict["Transformer/encoder_norm/scale"],
        params_dict["Transformer/encoder_norm/bias"]]
    pre_logic_weights = [
            params_dict["pre_logits/kernel"],
            params_dict["pre_logits/bias"]] if "pre_logits/kernel" in keys else []
    mlp_head_weights = [
        params_dict["head/kernel"],
        params_dict["head/bias"]]
    
    load_inf = {}
    for l in keras_model.layers:
        if l.name == "patch_embedding":
            try:
                l.set_weights(patch_emb_weights)
                load_inf[l.name] = "loaded"
            except: load_inf[l.name] = "mismatch"
    
        elif l.name == "add_cls_token":
            try:
                l.set_weights(cls_token_weights)
                load_inf[l.name] = "loaded - imagenet"
            except: load_inf[l.name] = "not loaded - mismatch"
    
        elif l.name == "position_embedding":
            try:
                l.set_weights(position_emb_weights)
                load_inf[l.name] = "loaded - imagenet"
            except: load_inf[l.name] = "not loaded - mismatch"
    
        elif l.name.startswith("transformer_block_"):
            i = int(l.name[l.name.rfind("_")+1:])
            try:
                weights = [w.reshape(s.shape) for w,s in zip(transformer_block_weights[i], l.weights)]
                l.set_weights(weights)
                load_inf[l.name] = "loaded - imagenet"
            except: load_inf[l.name] = "not loaded - mismatch"
    
        elif l.name == "layer_norm":
            try:
                l.set_weights(layer_norm_weights)
                load_inf[l.name] = "loaded - imagenet"
            except: load_inf[l.name] = "not loaded - mismatch"
    
        elif l.name == "pre_logits":
            try:
                l.set_weights(pre_logic_weights)
                load_inf[l.name] = "loaded - imagenet"
            except: 
                if pre_logic_weights == []:
                    load_inf[l.name] = "not loaded - not found"
                else:
                    load_inf[l.name] = "not loaded - mismatch"
    
        elif l.name == "mlp_head":
            try:
                l.set_weights(mlp_head_weights)
                load_inf[l.name] = "loaded - imagenet"
            except: load_inf[l.name] = "not loaded - mismatch"
    
    keras_model.load_weights_inf = load_inf


def get_imagenet2012_classes():
    """Get the dict of ImageNet 2012 classes."""
    classes_txt = utils.get_file(FNAME_CLASSES_IN2012, origin=BASIC_URL+FNAME_CLASSES_IN2012, cache_subdir="classes")
    with open(classes_txt, encoding="utf-8") as f:
        classes = {i:l.strip() for i,l in enumerate(f.readlines())}
    return classes


def read_img(file_path, resize=None):
    """Read an image and scale it to [-1,1].
    args:
        - file_path: The image file path
        - resize: The target size of image
    return:
        - An np array with values between 0 and 1
    
    """
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, resize)
    return img/127.5-1