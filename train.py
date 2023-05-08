# coding=gbk

import os
from tqdm import tqdm
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from model import *
from data import *
from vit_keras import vit


#---------------------------------设置参数-------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CONFIG_B = {    # ViT-B配置
    "hidden_dim":768,                              # patch嵌入维度
    "liner_dim":3072,                              # mlp线性变换维度
    "atten_heads":12,                              # 注意力头数
    "encoder_depth":12,                            # 编码器堆叠层数
    }

CONFIG_L = {    # ViT-L配置
    "hidden_dim":1024,     
    "liner_dim":4096,            
    "atten_heads":16,      
    "encoder_depth":24,    
    }

MODEL_CONFIG = CONFIG_B                            # 配置模型规模
WEIGHT_CONFIG = "imagenet21k"                      # 配置预训练权重
                                                   #"imagenet21k"包括pre_logits层权重，类别数：21843，输入尺寸为224
                                                   #"imagenet21k+imagenet2012" 没有pre_logits层权重，类别数：1000，输入尺寸为384

IMAGE_SIZE = 224                                   # 图片大小
PATCH_SIZE = 32                                    # patch大小
NUM_CLASSES = 10                                   # 输出类别
DROPOUT_RATE = 0.1                                 # dropout概率
ACTIVATION = "softmax"                             # 输出头激活函数
PRE_LOGITS = True                                  # 是否插入pre_logits层

LABEL_SMOOTH = 0.1                                 # 标签平滑系数
LEARNING_RATE = 1e-5                               # 初始学习率
BATCH_SIZE = 32                                    # 训练和验证的批大小
EPOCHS = 100                                       # 训练轮数
STEPS_PER_EPOCH = 40000//BATCH_SIZE                # 每轮训练的batch数
VALIDATION_STEPS = 10000//BATCH_SIZE               # 验证的batch数
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
DATA_PATH = "datasets/cifar-10"
SAVE_PATH = f"save_models/vit-{'b' if MODEL_CONFIG==CONFIG_B else 'l'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.h5"
PRE_TRAINED_PATH = f"pretrained/ViT-{'B' if MODEL_CONFIG==CONFIG_B else 'L'}_{PATCH_SIZE}_{WEIGHT_CONFIG}.npz"
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
train_data_gen, valid_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'train', 
    resize = (IMAGE_SIZE,IMAGE_SIZE)
    )
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
vit = ViT(
    image_size = IMAGE_SIZE,
    patch_size = PATCH_SIZE,
    num_classes = NUM_CLASSES, 
    dropout_rate = DROPOUT_RATE,
    activation = ACTIVATION,
    pre_logits = PRE_LOGITS,
    **MODEL_CONFIG
    )

# 加载预训练权重
vit.load_pretrained_weights(PRE_TRAINED_PATH)

vit.compile(
    optimizer=SGD(LEARNING_RATE, momentum=0.9),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['acc']
    )
vit.summary()
#-----------------------------------------------------------------------------


#--------------------------------训练和保存-------------------------------------
best_loss, _ = vit.evaluate(valid_data_gen, steps=VALIDATION_STEPS)
stop = 0
for e in range(EPOCHS):
    e_loss, e_acc = 0, 0
    for b in tqdm(range(STEPS_PER_EPOCH), f"training"):
        (x_batch, y_batch) = next(train_data_gen)
        loss, acc = vit.train_on_batch(x_batch, y_batch)
        e_loss += loss
        e_acc += acc
    print(f"[epoch:{e}]\
        \t[loss:{round(e_loss/STEPS_PER_EPOCH,4)}]\
        \t[acc:{round(e_acc/STEPS_PER_EPOCH,4)}]\
        \t[lr:{K.get_value(vit.optimizer.lr)}]") 
    loss, acc = vit.evaluate(valid_data_gen, steps=VALIDATION_STEPS)
    if loss < best_loss:
        best_loss = loss
        stop = 0
        vit.save_weights(SAVE_PATH)
        print(f"model saved")
    else:
        stop += 1
        if stop >= 3:
            print(f"early stop with eopch {e}")
            break
    lr = K.get_value(vit.optimizer.lr)
    K.set_value(vit.optimizer.lr, lr*0.9)
    print("-"*100)
#-----------------------------------------------------------------------------