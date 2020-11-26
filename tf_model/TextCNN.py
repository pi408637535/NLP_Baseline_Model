from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import tensorflow as tf


class TextCNN(Model):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        if config.embedding_pretrained is not None:
            self.embedding = layers.Embedding(config.n_vocab, config.embed_size, weights=[config.embedding_matrix], input_length=39)
        else:
            self.embedding = layers.Embedding(config.n_vocab, config.embed_size)

        self.convs = [layers.Conv2D(config.filter_num,
                                kernel_size=(filter, config.embed_size)) for filter in config.filters]
        #sentence_max_size=59
        self.max_pools = [layers.MaxPooling1D(pool_size=config.sentence_max_size - filter + 1, data_format="channels_last")
                          for filter in config.filters]
        self.dropout = layers.Dropout(rate=0.5)
        self.fc = layers.Dense(config.label_num)

    @staticmethod
    def conv_max_pool(text, convs, max_pools):
        conv1d_text = [tf.squeeze(conv(text), axis=-2) for conv in convs]

        # tensorflow 数据的位置与pytorch不一样
        max_pool_text = [tf.squeeze(max_pool(conv1d_text[index]), axis=-2) for index, max_pool in enumerate(max_pools)]
        return max_pool_text

    def call(self, inputs, training=None, mask=None):
        #inputs:batch,seq
        embed = self.embedding(inputs)

        #embed: batch, seq, embed
        embed = tf.expand_dims(embed, axis=-1)
        #embed: batch,seq,embed,-1

        conv_max_pool_res = self.conv_max_pool(embed, self.convs, self.max_pools)
        # max_pool_0: batch, filter
        total_max_pool = tf.concat(conv_max_pool_res, axis=-1)

        # total_max_pool : batch, filter_num * len(config.filters)
        total_max_pool = self.dropout(total_max_pool)

        return self.fc(total_max_pool)


