import torch as t
import torch.nn as nn
from torch_model.BasicModule import BasicModule

'''
# orininal textcnn
class TextCNN(BasicModule):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size)

        #2,3,4
        self.conv2d_0 = nn.Conv2d(1, config.filter_num, kernel_size =(config.filters[0], config.embed_size))
        self.conv2d_1 = nn.Conv2d(1, config.filter_num, kernel_size =(config.filters[1], config.embed_size))
        self.conv2d_2 = nn.Conv2d(1, config.filter_num, kernel_size =(config.filters[2], config.embed_size))

        self.max_pool_0 = nn.MaxPool1d(kernel_size=config.sentence_max_size - config.filters[0] + 1)
        self.max_pool_1 = nn.MaxPool1d(kernel_size=config.sentence_max_size - config.filters[1] + 1)
        self.max_pool_2 = nn.MaxPool1d(kernel_size=config.sentence_max_size - config.filters[2] + 1)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear( len(config.filters) * config.filter_num,  config.label_num)

    def forward(self, text):
        #text: batch,seq
        embed = self.embedding(text)

        #embed： batch,seq,embed
        embed = t.unsqueeze(embed, dim= 1)

        #embed: batch,1,seq, embed

        conv1d_0 = self.conv2d_0(embed)
        conv1d_1 = self.conv2d_1(embed)
        conv1d_2 = self.conv2d_2(embed)

        #conv1d_0: batch,filter_num,seq - 2 + 1, 1
        conv1d_0 = t.squeeze(conv1d_0, dim = -1)
        conv1d_1 = t.squeeze(conv1d_1, dim=-1)
        conv1d_2 = t.squeeze(conv1d_2, dim=-1)

        #conv1d_0: batch, filter_num, seq
        max_pool_0 = self.max_pool_0(conv1d_0)
        max_pool_1 = self.max_pool_1(conv1d_1)
        max_pool_2 = self.max_pool_2(conv1d_2)

        #max_pool_0:batch,filter_num,1
        max_pool_0 = t.squeeze(max_pool_0, dim = -1)
        max_pool_1 = t.squeeze(max_pool_1, dim=-1)
        max_pool_2 = t.squeeze(max_pool_2, dim=-1)

        #max_pool_0: batch, filter
        total_max_pool = t.cat([max_pool_0, max_pool_1, max_pool_2], dim= -1)

        # total_max_pool : batch, filter_num * len(config.filters)
        total_max_pool = self.dropout(total_max_pool)


        return self.fc(total_max_pool)
'''


class TextCNN(BasicModule):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size)

        self.convs = [ nn.Conv2d(1, config.filter_num,
                                 kernel_size =(filter, config.embed_size)) for filter in config.filters ]

        self.max_pools = [ nn.MaxPool1d(kernel_size=config.sentence_max_size - filter + 1)
                           for filter in config.filters ]

        self.fc = nn.Linear(len(config.filters) * config.filter_num, config.label_num)

    @staticmethod
    def conv_max_pool(text, convs, max_pools):
        conv1d_text = [ t.squeeze(conv(text), dim=-1) for conv in convs]
        max_pool_text = [ t.squeeze(max_pool(conv1d_text[index]), dim=-1) for index,max_pool in enumerate(max_pools) ]
        return max_pool_text


    def forward(self, text):
        # text: batch,seq
        embed = self.embedding(text)

        # embed： batch,seq,embed
        embed = t.unsqueeze(embed, dim=1)

        conv_max_pool_res = self.conv_max_pool(embed, self.convs, self.max_pools)
        # max_pool_0: batch, filter
        total_max_pool = t.cat(conv_max_pool_res, dim=-1)

        # total_max_pool : batch, filter_num * len(config.filters)
        total_max_pool = self.dropout(total_max_pool)

        return self.fc(total_max_pool)
