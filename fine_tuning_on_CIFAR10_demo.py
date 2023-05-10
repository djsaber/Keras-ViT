from tqdm import tqdm
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras import backend as K
from data import cifar_10_data_gen
from keras_vit.vit import ViT_B32, ViT


#---------------------------------设置参数-------------------------------------
WEIGHT_CONFIG = "imagenet21k"                      # 配置预训练权重 "imagenet21k" 或者 "imagenet21k+imagenet2012"
NUM_CLASSES = 10                                   # 输出类别
DROPOUT_RATE = 0.1                                 # dropout概率
ACTIVATION = "softmax"                             # 输出头激活函数

LABEL_SMOOTH = 0.1                                 # 标签平滑系数
LEARNING_RATE = 1e-3                               # 初始学习率
BATCH_SIZE = 16                                    # 训练和验证的批大小
EPOCHS = 100                                       # 训练轮数
STEPS_PER_EPOCH = 40000//BATCH_SIZE                # 每轮训练的batch数
VALIDATION_STEPS = 10000//BATCH_SIZE               # 验证的batch数
TEST_STEPS = 10000//BATCH_SIZE                     # 测试的batch数
#-----------------------------------------------------------------------------


#---------------------------------设置路径-------------------------------------
DATA_PATH = "datasets/cifar-10"
SAVE_PATH = f"save_models/"
#-----------------------------------------------------------------------------


#---------------------------------搭建模型-------------------------------------
vit = ViT_B32(
    num_classes = NUM_CLASSES, 
    activation = ACTIVATION,
    dropout_rate = DROPOUT_RATE,
    weights=WEIGHT_CONFIG,
    )

vit.summary()
vit.loading_summary()

vit.compile(
    optimizer=SGD(LEARNING_RATE, momentum=0.9),
    loss=CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=['acc']
    )
#-----------------------------------------------------------------------------


#--------------------------------加载数据集-------------------------------------
train_data_gen, valid_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'train', 
    resize = vit.image_size
    )
test_data_gen = cifar_10_data_gen(
    path = DATA_PATH, 
    batch_size = BATCH_SIZE, 
    data = 'test', 
    resize = vit.image_size
    )
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
        vit.save_weights(SAVE_PATH+vit.name+".h5")
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


#----------------------------------测试----------------------------------------
vit.evaluate(
    test_data_gen, 
    steps=TEST_STEPS
    )
#-----------------------------------------------------------------------------
