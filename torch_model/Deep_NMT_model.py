import torch as t
import torch.nn as nn
from torch_model.BasicModule import BasicModule
import torch.nn.functional as F

class DeepNMT(BasicModule):
    def __init__(self, config):
        super(DeepNMT, self).__init__()
        self.source_embedding = nn.Embedding(config.source_vocab_size, config.embed_size)
        self.encoder = nn.LSTM(input_size=config.embed_size,
                               hidden_size=config.lstm_size,num_layers=4,
                               batch_first=True)

        self.decoder_embedding = nn.Embedding(config.target_vocab_size, config.embed_size)
        self.decoder = nn.LSTM(input_size=config.embed_size,
                               hidden_size=config.lstm_size, num_layers=4,
                               batch_first=True)

        self.max_length = config.max_length
        self.fc = nn.Linear(config.lstm_size, config.vacab_size)

    def forward(self, source_text, target_text_input):
        #source_text:batch,seq
        source_embed = self.source_embedding(source_text)

        #source_output: batch, seq, hid
        #hid: layer * dir, batch ,hid
        source_output,_ = self.encoder(source_embed)

        decode_hid = _[0]
        c_hid  = _[1]
        target_embed = self.decoder_embedding(target_text_input)
        #target_embed: batch, seq, embed
        # <sos> <y_1...> <eos>
        res = []
        for i in range(self.max_length):
            output,(h_n, c_n) = self.decoder(target_embed[:, i, :].unsqueeze(dim=1), (decode_hid, c_hid) )
            #output:batch, 1, hid * dir
            # layer * direction, batch, hidden
            decode_hid, c = h_n, c_n
            res.append(output)

        res = t.cat(res, dim=1)
        #res:batch, seq, hidden
        return self.fc(res) #batch, seq, vocab



class LMLoss(nn.Module):
    def __init__(self):
        super(LMLoss, self).__init__()

    def forward(self, target, output, mask = None):
        #target: batch, seq
        #output: batch, seq, vocab
        target = t.unsqueeze(target, dim=-1)

        #target: batch, seq, 1
        output = F.softmax(output, dim = -1)
        output = t.gather(output, dim=-1, index = target)

        #output: batch, seq, 1
        output = t.squeeze(output, dim = -1)
        batch = output.shape[0]
        if mask:
            #output: batch, seq
            output = output * mask

            total_words_num = t.sum(mask)


            return t.sum(output) / total_words_num / batch
        else:
            return t.sum(output) / batch

