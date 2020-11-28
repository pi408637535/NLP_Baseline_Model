import torch as t
import torch.nn as nn
from torch_model.BasicModule import BasicModule

class DeepNMT(BasicModule):
    def __init__(self, config):
        super(DeepNMT, self).__init__()
        self.source_embedding = nn.Embedding(config.source_vocab_size, config.source_embedding_size)
        self.encoder = nn.LSTM(input_size=config.source_embedding_size,
                               hidden_size=config.lstm_size,num_layers=4,
                               batch_first=True)

        self.decoder_embedding = nn.Embedding(config.target_vocab_size, config.target_embedding_size)
        self.decoder = nn.LSTM(input_size=config.target_embedding_size,
                               hidden_size=config.lstm_size,num_layers=4,
                               batch_first=True)

        self.max_length = config.max_length

    def forward(self, source_text, target_text):
        #source_text:batch,seq
        source_embed = self.source_embedding(source_text)

        #source_output: batch, seq, hid
        #hid: layer * dir, batch ,hid
        source_output,_ = self.encoder(source_embed)

        decode_hid = _[0]
        c = t.ones()
        target_embed = self.decoder_embedding(target_text)
        # <sos> <y_1...> <eos>
        for i in range(self.max_length):
             self.decoder(target_embed[:, i, :], (decode_hid, c) )



