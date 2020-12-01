# —*- coding: utf-8 -*-

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_size', type=int, default=300, help="embedding size of word embedding")
    parser.add_argument("--epoch",type=int,default=200,help="epoch of training")
    parser.add_argument("--cuda",type=bool,default=True,help="whether use gpu")
    parser.add_argument("--gpu",type=int,default=0,help="whether use gpu")
    parser.add_argument("--label_num",type=int,default=2,help="the label number of samples")
    parser.add_argument("--learning_rate",type=float,default=0.0005,help="learning rate during training")
    parser.add_argument("--batch_size",type=int,default=50,help="batch size during training")
    parser.add_argument("--filter_num",type=int,default=100,help="filter num of conv")
    parser.add_argument("--filters",type=str,default="3,4,5",help="filters size of conv")
    parser.add_argument("--dropout",type=float,default=0.5,help="dropout of training")
    parser.add_argument("--seed",type=int,default=1,help="seed of random")
    parser.add_argument("--l2",type=float,default=0.004,help="l2 weight")
    parser.add_argument("--use_pretrained_embed",type=bool,default=True,help="whether use the pretrained embedding")
    parser.add_argument("--embedding_type",type=str,default="word2vec",help="word2vec or glove or no")
    parser.add_argument("--sentence_max_size", type=int, default=10, help="sentence_max_size")


    # seq2seq
    parser.add_argument("--source_vocab_size", type=int, default=30000, help="source_vocab_size")
    parser.add_argument("--target_vocab_size", type=int, default=30000, help="source_vocab_size")
    parser.add_argument("--source_length", type=int, default=100, help="source_length")
    parser.add_argument("--target_length", type=int, default=100, help="target_length")
    parser.add_argument("--lstm_size", type=int, default=256, help="lstm_size")
    parser.add_argument("--max_length", type=int, default=100, help="max_length")

    parser.embedding_pretrained = None
    return parser.parse_args()
