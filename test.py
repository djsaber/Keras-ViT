# coding=gbk

import os
from model import *
from data import *


#---------------------------------设置参数-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CONFIG_B = {
    "hidden_dim":768,                              # patch嵌入维度
    "liner_dim":3072,                              # mlp线性变换维度
    "atten_heads":12,                              # 注意力头数
    "encoder_depth":12,                            # 编码器堆叠层数
    }

CONFIG_L = {    
    "hidden_dim":1024,     
    "liner_dim":4096,            
    "atten_heads":16,      
    "encoder_depth":24,    
    }

MODEL_CONFIG = CONFIG_B                            # 配置模型规模
WEIGHT_CONFIG = "imagenet21k"                      # 配置预训练权重

IMAGE_SIZE = 224                                   # 图片大小
PATCH_SIZE = 32                                    # patch大小
NUM_CLASSES = 10                                   # 输出类别
DROPOUT_RATE = 0.1,                                # dropout概率
ACTIVATION = "softmax"                             # 输出头激活函数
PRE_LOGITS = True                                  # 是否插入pre_logits层

BATCH_SIZE = 10                                    # 测试的批大小
STEPS=10000//BATCH_SIZE                            # 测试集的batch数
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
SAVE_PATH = f"save_models/vit-{'b' if MODEL_CONFIG==CONFIG_B else 'l'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.h5"
DATA_PATH = "datasets/cifar-10"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
test_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'test', 
    resize = (IMAGE_SIZE,IMAGE_SIZE)
    )
#-----------------------------------------------------------------------------


#---------------------------------加载模型-------------------------------------
vit = ViT(
    image_size = IMAGE_SIZE,
    patch_size = PATCH_SIZE,
    num_classes = NUM_CLASSES, 
    dropout_rate = DROPOUT_RATE,
    activation = ACTIVATION,
    pre_logits = PRE_LOGITS,
    **MODEL_CONFIG)
vit.load_weights(SAVE_PATH)
vit.summary()
#-----------------------------------------------------------------------------


#----------------------------------测试模式-------------------------------------
vit.compile(
    loss='categorical_crossentropy', 
    metrics=['acc']
    )
vit.evaluate(
    test_data_gen, 
    steps=STEPS
    )
#-----------------------------------------------------------------------------