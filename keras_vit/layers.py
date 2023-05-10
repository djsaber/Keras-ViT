from keras import layers
from keras import activations
from keras import backend as K

class PatchEmbedding(layers.Layer):
    """patch embedding layer"""
    def __init__(self, patch_size, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.conv = layers.Conv2D(self.hidden_dim, self.patch_size, self.patch_size)
        self.flatten = layers.Reshape((-1, self.hidden_dim))
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.flatten(x)
        return x


class AddCLSToken(layers.Layer):
    """add class token layer"""
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token_weight",
            shape=(1, 1, self.hidden_dim),
            initializer="zero",
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):   
        x = K.tile(self.class_token, [K.shape(inputs)[0],1,1])
        x = K.concatenate([x, inputs], axis=1)
        return x


class AddPositionEmbedding(layers.Layer):
    """add position encoding layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.pe = self.add_weight(
            name="position_embedding",
            shape=(1, input_shape[1], input_shape[2]),
            initializer='random_normal',
            trainable=True
            )
        return super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pe


class MultiHeadAttention(layers.Layer):
    """multi head self attention layer"""
    def __init__(
        self, 
        heads,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.heads = heads

    def build(self, input_shape):
        self.dk = K.sqrt(K.cast(input_shape[-1]//self.heads, dtype=K.tf.float32))
        self.q_dense = layers.Dense(input_shape[-1], name="query")
        self.k_dense = layers.Dense(input_shape[-1], name="key")
        self.v_dense = layers.Dense(input_shape[-1], name="value")
        self.o_dense = layers.Dense(input_shape[-1], name="combine_out")
        return super().build(input_shape)

    def call(self, inputs):
        q = self.q_dense(inputs)
        k = self.k_dense(inputs)
        v = self.v_dense(inputs)
        q = K.concatenate(K.tf.split(q, self.heads, axis=-1), axis=0)
        k = K.concatenate(K.tf.split(k, self.heads, axis=-1), axis=0)
        v = K.concatenate(K.tf.split(v, self.heads, axis=-1), axis=0)
        qk = K.tf.matmul(q, k, transpose_b=True)  
        qk = activations.softmax(qk / self.dk)
        qkv = K.tf.matmul(qk, v)
        qkv = K.concatenate(K.tf.split(qkv, self.heads, axis=0), axis=-1)
        return self.o_dense(qkv)


class MLPBlock(layers.Layer):
    """mlp block"""
    def __init__(self, liner_dim, dropout_rate, **kwargs):
        self.liner_dim = liner_dim
        self.dropout_rate = dropout_rate
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.dropout = layers.Dropout(self.dropout_rate)
        self.liner_1 = layers.Dense(self.liner_dim, activations.gelu)
        self.liner_2 = layers.Dense(input_shape[-1])
        return super().build(input_shape)

    def call(self, inputs):
        h = self.liner_1(inputs)
        h = self.dropout(h)
        h = self.liner_2(h)
        h = self.dropout(h)
        return h


class TransformerEncoder(layers.Layer):
    """transformerd encoder block"""
    def __init__(
        self, 
        liner_dim,
        atten_heads,
        dropout_rate,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.liner_dim = liner_dim
        self.atten_heads = atten_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):    
        self.multi_head_attens = MultiHeadAttention(
            name='multi_head_attention_layer',
            heads=self.atten_heads, 
            )
        self.mlp_block = MLPBlock(
            name='mlp_block_layer',
            liner_dim=self.liner_dim,
            dropout_rate = self.dropout_rate
            )
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, inputs):
        x = self.layer_norm_1(inputs)
        x = self.multi_head_attens(x)
        x = self.dropout(x)
        x = x + inputs
        y = self.layer_norm_2(x)
        y = self.mlp_block(y)
        return x + y