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
        self.fc = nn.Linear(config.lstm_size, config.target_vocab_size)

    def forward(self, source_text, target_text_input, mode = "train"):
        #source_text:batch,seq
        source_embed = self.source_embedding(source_text)

        #source_output: batch, seq, hid
        #enc_hidden: h: layer * dir, batch ,hid
        source_output,enc_hidden = self.encoder(source_embed)

        #target_embed: batch,seq,embed
        target_embed = self.decoder_embedding(target_text_input)
        if mode == "train":
            res,_ = self.decoder(target_embed, enc_hidden)
        else:
            res = []
            dec_prev_hidden = enc_hidden
            for i in range(self.max_length):
                output, dec_hidden = self.decoder(target_embed[:, i, :].unsqueeze(dim=1), dec_prev_hidden)
                # output:batch, 1, hid * dir
                # layer * direction, batch, hidden
                dec_prev_hidden = dec_hidden
                res.append(output)

            #res: batch, max_length, hidden
            res = t.cat(res, dim=1)

        #target_embed: batch, seq, embed
        # <sos> <y_1...> <eos>

        #res:batch, seq, hidden
        return self.fc(res) #batch, seq, vocab



class LMLoss(nn.Module):
    def __init__(self):
        super(LMLoss, self).__init__()

    def forward(self, target, output, reduce = False):
        #target: batch, seq
        #output: batch, seq, vocab
        target = t.unsqueeze(target, dim=-1)

        # target: batch, seq, 1
        output = F.log_softmax(output, dim = -1)
        output = t.gather(output, dim=-1, index = target)

        #output: batch, seq, 1
        output = t.squeeze(output, dim = -1)
        batch = output.shape[0]

        #ouput: batch, seq
        mask = t.squeeze(target != 0, dim=-1)
        if reduce:
            loss = t.sum(output * mask.float()) / t.sum(mask.float()) / batch
        else:
            loss = t.sum(output * mask.float()) / t.sum(mask.float())
        return -loss # Negative



