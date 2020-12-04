import torch as t
import torch.nn as nn
from torch_model.BasicModule import BasicModule
import torch.nn.functional as F


class AttentionNMT(BasicModule):
    def __init__(self, config):
        super(AttentionNMT, self).__init__()
        self.source_embedding = nn.Embedding(config.source_vocab_size, config.embed_size)
        self.encoder = nn.GRU(input_size=config.embed_size,
                               hidden_size=config.lstm_size,num_layers=1,bidirectional=True,
                               batch_first=True)

        self.decoder_embedding = nn.Embedding(config.target_vocab_size, config.embed_size)
        self.decoder = nn.GRU(input_size=config.lstm_size * 2 + config.embed_size,
                               hidden_size=config.lstm_size , num_layers=1,
                               batch_first=True)

        # (encoder) 2*hidden + (decoder) 1*hidden
        self.attention_fc_1 = nn.Linear(3 * config.lstm_size, 3 * config.lstm_size)
        self.attention_fc_2 = nn.Linear(3 * config.lstm_size, 1)


        self.class_fc_1 = nn.Linear(config.lstm_size * 2 + config.embed_size + config.lstm_size, config.lstm_size * 2)
        self.class_fc_2 = nn.Linear(config.lstm_size * 2, config.target_vocab_size)

        self.max_length = config.max_length

    def attention_forward(self, input_source, dec_prev_hidden, enc_output):
        '''
        :param input_source: batch,hidden
        :param dec_prev_hidden: batch, hidden
        :param enc_output: batch,seq, hidden * 2
        :return:
        '''
        dec_prev_hidden = dec_prev_hidden.squeeze(dim=0)
        dec_prev_hidden = dec_prev_hidden.unsqueeze(dim=1).repeat(1, 100, 1)

        #score:batch,seq, 1
        score = self.attention_fc_2( self.attention_fc_1( t.cat([dec_prev_hidden, enc_output], dim=-1) ) )
        score = F.softmax(score, dim= -1)

        #score:batch,seq,1
        context = t.sum(score * enc_output, dim=1) #context:batch,2 * hidden
        input_source = t.unsqueeze(input_source, dim=1)

        context = t.unsqueeze(context, dim=1)

        #context: batch, 1, hidden * 2
        #input_source:batch, 1, hidden
        dec_output, dec_hidden = self.decoder(t.cat([input_source, context], dim=-1))

        return context,dec_output,dec_hidden



    def forward(self, source_text, target_text_input, mode = "train", is_gpu = False):
        #source_text:batch,seq
        source_embed = self.source_embedding(source_text)

        #source_output: batch, seq, hidden * 2
        #enc_hidden: h: layer * dir, batch, hidden ： 2, batch, hidden
        source_output,enc_hidden = self.encoder(source_embed)

        #target_embed: batch,seq,embed
        target_embed = self.decoder_embedding(target_text_input)

        batch, seq, _ = source_output.shape
        encoder_hidden_size = source_output.shape[2]
        decoder_hidden_size = enc_hidden.shape[2]
        self.atten_outputs = t.autograd.Variable(t.zeros(batch, seq, encoder_hidden_size ))
        self.dec_outputs = t.autograd.Variable(t.zeros(batch, seq, decoder_hidden_size))

        if is_gpu:
            self.atten_outputs = self.atten_outpus.cuda()
            self.dec_outputs = self.dec_outputs.cuda()

        if mode == "train":

            dec_prev_hidden = enc_hidden[0].unsqueeze(dim=0)
            #dec_prev_hidden: 1, batch, hidden

            for i in range(self.max_length):
                input_source = target_embed[:, i, :]

                #atten_output:batch, 1, hidden * 2
                atten_output, dec_output, dec_hidden = self.attention_forward( input_source, dec_prev_hidden, source_output)

                self.atten_outputs[:, i] = atten_output.squeeze(dim=1)
                self.dec_outputs[:, i] = dec_hidden.squeeze(dim=1)
                dec_prev_hidden = dec_hidden

            #class_input:batch, seq, hidden * 4
            class_input = t.cat([target_embed, self.atten_outputs, self.dec_outputs], dim=2)

            #outs:batch,seq,vocab
            outs = self.class_fc_2( F.relu( self.class_fc_1(class_input)) )
        else:
            dec_prev_hidden = enc_hidden[0].unsqueeze(dim=0)
            # dec_prev_hidden: 1, batch, hidden

            for i in range(self.max_length):
                input_source = target_embed[:, i, :]

                atten_output, dec_output, dec_hidden = self.attention_forward(input_source, dec_prev_hidden, source_output)

                self.atten_outputs[:, i] = atten_output.squeeze(dim=1)
                self.dec_outputs[:, i] = dec_hidden.squeeze(dim=1)
                dec_prev_hidden = dec_hidden

            # class_input:batch, seq, hidden * 4
            class_input = t.cat([target_embed, self.atten_outputs, self.dec_outputs], dim=2)

            # outs:batch,seq,vocab
            preds = self.class_fc_2(F.relu(self.class_fc_1(class_input)))

            #outs:batch,seq,1
            outs = t.argmax(preds, dim=-1)
            outs = outs.squeeze().cpu().numpy()

        return outs


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



