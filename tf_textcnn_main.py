
from data.TF_MR_Dataset import TF_MR_Dataset
import tensorflow as tf
from tensorflow.keras import layers
from tf_model.TextCNN import  TextCNN

import config as argumentparser

config = argumentparser.ArgumentParser()
config.filters = list(map(int,config.filters.split(",")))


if __name__ == '__main__':
    tf_mr_data = TF_MR_Dataset()
    train_X, train_Y, test_X, test_Y = tf_mr_data.get_data()

    train_X = tf.convert_to_tensor(train_X, dtype=tf.int32)
    train_Y = tf.convert_to_tensor(train_Y, dtype=tf.int32)
    test_X = tf.convert_to_tensor(test_X, dtype=tf.int32)
    test_Y = tf.convert_to_tensor(test_Y, dtype=tf.int32)

    AUTO = tf.data.experimental.AUTOTUNE

    batch_size = 32
    train_set = (tf.data.Dataset.from_tensor_slices(
        (train_X,train_Y))
        .repeat()
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(AUTO))

    test_set = (tf.data.Dataset.from_tensor_slices(
        (test_X,test_Y))
        .shuffle(2048)
        .batch(batch_size)
        .prefetch(AUTO))

    vocab_size = tf_mr_data.get_vocab_size()
    embed_size = 300
    embedding_matrix = tf_mr_data.get_word2vec()
    num_hiddens = 100



    if config.use_pretrained_embed:
        config.embedding_pretrained = True
        config.embedding_matrix = embedding_matrix
    else:
        config.embedding_pretrained = False
    config.n_vocab = tf_mr_data.get_vocab_size()

    model = TextCNN(config)
    # model.layers[0].trainable = False
    model.build(input_shape=(None, 59))
    model.summary()