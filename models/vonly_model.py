import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class AttentiveCNN(nn.Module):

    def __init__(self):
        super(AttentiveCNN, self).__init__()
        # ResNet-152 backend
        resnet = models.resnet152(pretrained=True )
        modules = list(resnet.children())[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential(*modules) # last conv feature
        
        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        
    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''  
        # Last conv layer feature map
        A = self.resnet_conv(images)
        # a^g, average pooling feature map
        a_g = self.avgpool(A)
        v_g = a_g.view(a_g.size(0), -1)
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view(A.size(0), A.size(1), -1).transpose(1,2)
        
        return V, v_g

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.args = args
        self.drop_prob = args.drop_prob

        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            if args.dataset == 'data':
                self.vocab_size = args.vocab_size
                self.max_caption_len = 31
            elif args.dataset == 'cn_data':
                self.vocab_size = args.cn_vocab_size
                self.max_caption_len = 15

        self.hidden_size = args.hidden_size

        self.cnn = AttentiveCNN()
        self.mlp = nn.Linear(self.hidden_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.mlp.bias.data.fill_(0)
        self.mlp.weight.data.uniform_(-initrange, initrange)

    def forward(self, train_data):
        images = train_data['images']
        img_num = train_data['img_num']
        captions = train_data['captions']
        lengths = train_data['lengths']

        batch_size = images.size(0)
        steps = captions.size(1)

        LMcriterion = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            LMcriterion = LMcriterion.cuda()
            states = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
            cs = torch.zeros(batch_size, steps, self.hidden_size).cuda()
            hiddens = Variable(torch.zeros(batch_size, 1, self.hidden_size)).cuda()
        else:
            states = Variable(torch.zeros(1, batch_size, self.hidden_size))
            cs = torch.zeros(batch_size, steps, self.hidden_size)
            hiddens = Variable(torch.zeros(batch_size, 1, self.hidden_size))

        targets = pack_padded_sequence(captions[:,1:], lengths, batch_first=True)[0]
        
        images = images.view(-1, 3, 224, 224)
        V, v_g = self.cnn(images)
        V = torch.sum(V.view(batch_size, -1, 49, 2048), dim=1)
        v_g = torch.sum(v_g.view(batch_size, -1, 2048), dim=1)

        for i in range(batch_size):
            V[i] = torch.div(V[i], img_num[i])
            v_g[i] = torch.div(v_g[i], img_num[i])

        for time in range(steps): 
            hiddens, states = self.core(time, V, v_g, captions, states, hiddens)
            cs[:, time, :] = hiddens.squeeze(1)
        
        scores = F.log_softmax(self.mlp(self.dropout(cs)), dim = 2)
        
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)[0]   
        if self.args.use_crossentropy:
            loss = LMcriterion(packed_scores, targets - 1) 
        else:
            loss = - packed_scores.gather(1, (targets - 1).unsqueeze(1)).squeeze(1) 
        loss = torch.mean(loss) 
        return loss

    def sampler(self, val_data):
        max_len = max_caption_len
        images = val_data['images']
        img_num = val_data['img_num']
        batch_size = images.size(0)
        images = images.view(-1, 3, 224, 224)
        V, v_g = self.cnn(images)
        V = torch.sum(V.view(batch_size, -1, 49, 2048), dim=1)
        v_g = torch.sum(v_g.view(batch_size, -1, 2048), dim=1)

        for i in range(batch_size):
            V[i] = torch.div(V[i], img_num[i])
            v_g[i] = torch.div(v_g[i], img_num[i])

        if torch.cuda.is_available():
            caption = Variable(torch.LongTensor(batch_size, 1).fill_(1)).cuda()
            states = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
            hiddens = Variable(torch.zeros(batch_size, 1, self.hidden_size)).cuda()
        else:
            caption = Variable(torch.LongTensor(batch_size, 1).fill_(1))
            states = Variable(torch.zeros(1, batch_size, self.hidden_size))
            hiddens = Variable(torch.zeros(batch_size, 1, self.hidden_size))

        sampled_ids = []

        for time in range(max_len):
            hiddens, states = self.core(time, V, v_g, caption, states, hiddens, val = True)
            scores = F.softmax(self.mlp(self.dropout(hiddens)), dim = 2)

            predicted = scores.max(2)[1] + 1
            caption = predicted

            sampled_ids.append(caption)

        sampled_ids = torch.cat(sampled_ids, dim = 1)

        return sampled_ids

class ShowTellCore(nn.Module):
    def __init__(self, args):
        super(ShowTellCore, self).__init__()
        self.img_feat_size = args.img_feat_size
        self.input_encoding_size = args.hidden_size
        self.hidden_size = args.hidden_size

        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.img_embed = nn.Linear(self.img_feat_size, self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.GRU = nn.GRU(self.input_encoding_size, self.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, time, V, v_g, captions, states, hiddens, val = False):
        
        if time == 0:
            x_t = self.img_embed(v_g).unsqueeze(1)
        else:
            if val:
                it = captions.clone().squeeze(1)
            else:
                it = captions[:, time-1].clone()
            x_t = self.embed(it).unsqueeze(1)

        x_t = x_t.permute(1,0,2)

        output, states = self.GRU(x_t, states)
        output = output.permute(1,0,2)

        return output, states

class ShowAttendTellCore(nn.Module):
    def __init__(self, args):
        super(ShowAttendTellCore, self).__init__()
        self.img_feat_size = args.img_feat_size
        self.input_encoding_size = args.hidden_size

        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.fc_I_v = nn.Linear(self.img_feat_size, self.input_encoding_size, bias = False)
        self.fc_I_F = nn.Linear(self.input_encoding_size, 1, bias = False)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.GRU = nn.GRU(self.input_encoding_size + self.img_feat_size, self.input_encoding_size, 1)

        self.dropout = nn.Dropout(0.2)#0.2
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        init.xavier_uniform_(self.fc_I_v.weight)
        init.xavier_uniform_(self.fc_I_F.weight) 

    def forward(self, time, V, v_g, captions, states, hiddens, val = False):

        content_v = self.fc_I_v(self.dropout(V)) + self.dropout(hiddens)
        z_v = self.fc_I_F(self.dropout(torch.tanh(content_v))).squeeze(2)
        alpha_v = F.softmax(z_v, dim = 1).unsqueeze(1)
        I = torch.bmm(alpha_v, V)
        
        if val:
            it = captions.clone().squeeze(1)
        else:
            it = captions[:, time].clone()
        x_t = self.embed(it).unsqueeze(1)
        
        input_t = torch.cat([x_t, I], dim = 2).permute(1,0,2)

        self.GRU.flatten_parameters()
        hiddens, states = self.GRU(input_t, states)
        hiddens = hiddens.permute(1,0,2)

        return hiddens, states

class AdaAtt_lstm(nn.Module):
    def __init__(self, args, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = args.hidden_size
        #self.rnn_type = args.rnn_type
        self.rnn_size = args.hidden_size
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob
        self.fc_feat_size = args.img_feat_size
        self.att_feat_size = args.img_feat_size
        self.att_hid_size = args.hidden_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = torch.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = torch.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = torch.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)
        '''
        state = torch.cat((torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                           torch.cat([_.unsqueeze(0) for _ in cs], 0)), 0).unsqueeze(1)
        '''
        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                 torch.cat([_.unsqueeze(0) for _ in cs], 0))

        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, args):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = args.hidden_size
        #self.rnn_type = args.rnn_type
        self.rnn_size = args.hidden_size
        self.drop_prob_lm = args.drop_prob
        self.att_hid_size = args.hidden_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = torch.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim = 1)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = torch.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaptAttendCore(nn.Module):
    def __init__(self, args, use_maxout=False):
        super(AdaptAttendCore, self).__init__()
        self.img_feat_size = args.img_feat_size
        self.rnn_size = args.hidden_size 
        self.input_encoding_size = args.hidden_size

        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.drop_prob_lm = args.drop_prob

        self.lstm = AdaAtt_lstm(args, use_maxout)
        self.attention = AdaAtt_attention(args)

        self.fc_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        self.ctx2att = nn.Linear(self.rnn_size, self.input_encoding_size)

    def forward(self, time, V, v_g, captions, states, hiddens, val = False):
        
        batch_size = V.size(0)

        fc_feats = self.fc_embed(v_g)
        att_feats_ = self.att_embed(V)
        att_feats = att_feats_.view(batch_size, 7, 7, att_feats_.size(2))
        p_att_feats = self.ctx2att(att_feats)
        
        if val:
            it = captions.clone().squeeze(1)
        else:
            it = captions[:, time].clone()
        x_t = self.embed(it)

        if time == 0:
            if torch.cuda.is_available():
                states = Variable(torch.zeros(2, 1, batch_size, self.input_encoding_size)).cuda()
            else:
                states = Variable(torch.zeros(2, 1, batch_size, self.input_encoding_size))

        h_out, p_out, state = self.lstm(x_t, fc_feats, states)
        atten_out_ = self.attention(h_out, p_out, att_feats, p_att_feats)
        atten_out = atten_out_.unsqueeze(1)

        return atten_out, states

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.rnn_size = args.hidden_size
        self.att_hid_size = args.hidden_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim = 1)                             # batch * att_size
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Att2in2Core(nn.Module):
    def __init__(self, args):
        super(Att2in2Core, self).__init__()
        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.input_encoding_size = args.hidden_size
        #self.rnn_type = args.rnn_type
        self.rnn_size = args.hidden_size
        #self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob
        self.img_feat_size = args.img_feat_size
        self.att_feat_size = args.img_feat_size
        self.att_hid_size = args.hidden_size

        #prepare input
        self.fc_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.ctx2att = nn.Linear(self.rnn_size, self.input_encoding_size)
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(args)

    def forward(self, time, V, v_g, captions, state, hiddens, val = False):

        batch_size = V.size(0)

        fc_feats = self.fc_embed(v_g)
        att_feats_ = self.att_embed(V)
        att_feats = att_feats_.view(batch_size, 7, 7, att_feats_.size(2))
        p_att_feats = self.ctx2att(att_feats)

        if val:
            it = captions.clone().squeeze(1)
        else:
            it = captions[:, time].clone()
        xt = self.embed(it)

        if time == 0:
            if torch.cuda.is_available():
                state = Variable(torch.zeros(2, 1, batch_size, self.input_encoding_size)).cuda()
            else:
                state = Variable(torch.zeros(2, 1, batch_size, self.input_encoding_size))

        att_res = self.attention(state[0][-1], att_feats, p_att_feats)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output_ = self.dropout(next_h)
        output = output_.unsqueeze(1)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))

        return output, state

class UpDownCore(nn.Module):
    def __init__(self, args, use_maxout=False):
        super(UpDownCore, self).__init__()
        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.img_feat_size = args.img_feat_size
        self.drop_prob_lm = args.drop_prob
        self.input_encoding_size = args.hidden_size
        self.rnn_size = args.hidden_size

        #prepare input
        self.fc_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.img_feat_size, self.rnn_size),
                                       nn.ReLU(),
                                       nn.Dropout(self.drop_prob_lm))
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.ctx2att = nn.Linear(self.rnn_size, self.input_encoding_size)

        self.att_lstm = nn.LSTMCell(self.input_encoding_size + self.rnn_size * 2, self.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(self.rnn_size * 2, self.rnn_size) # h^1_t, \hat v
        self.attention = Attention(args)

    def forward(self, time, V, v_g, captions, state, hiddens, val = False):
        
        batch_size = V.size(0)

        fc_feats = self.fc_embed(v_g)
        att_feats_ = self.att_embed(V)
        att_feats = att_feats_.view(batch_size, 7, 7, att_feats_.size(2))
        p_att_feats = self.ctx2att(att_feats)

        if val:
            it = captions.clone().squeeze(1)
        else:
            it = captions[:, time].clone()
        xt = self.embed(it)

        if time == 0:
            if torch.cuda.is_available():
                state = Variable(torch.zeros(2, 2, batch_size, self.input_encoding_size)).cuda()
            else:
                state = Variable(torch.zeros(2, 2, batch_size, self.input_encoding_size))

        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output_ = F.dropout(h_lang, self.drop_prob_lm, self.training)
        output = output_.unsqueeze(1)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class AllImgCore(nn.Module):
    def __init__(self, args):
        super(AllImgCore, self).__init__()
        if args.lower:
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_size = args.vocab_size

        self.input_encoding_size = args.hidden_size
        self.rnn_type = 'lstm'
        self.rnn_size = args.hidden_size
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob
        self.fc_feat_size = args.img_feat_size

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))

        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size) # feature to rnn_size

        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, time, V, v_g, captions, state, hiddens, val = False):
        
        batch_size = V.size(0)

        if val:
            it = captions.clone().squeeze(1)
        else:
            it = captions[:, time].clone()
        xt = self.embed(it)
        
        if time == 0:
            image_map = self.linear(v_g).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
            state = (image_map, image_map)

        self.rnn.flatten_parameters()
        output_, state = self.rnn(torch.cat([xt, v_g], 1).unsqueeze(0), state)
        output = output_.permute(1,0,2)

        return output, state

class ShowTellModel(BasicModel):
	def __init__(self, args):
		super(ShowTellModel, self).__init__(args)
		self.core = ShowTellCore(args)

class ShowAttendTellModel(BasicModel):
	def __init__(self, args):
		super(ShowAttendTellModel, self).__init__(args)
		self.core = ShowAttendTellCore(args)

class AdaptAttendModel(BasicModel):
	def __init__(self, args):
		super(AdaptAttendModel, self).__init__(args)
		self.core = AdaptAttendCore(args)

class Att2in2Model(BasicModel):
	def __init__(self, args):
		super(Att2in2Model, self).__init__(args)
		self.core = Att2in2Core(args)

class UpDownModel(BasicModel):
	def __init__(self, args):
		super(UpDownModel, self).__init__(args)
		self.core = UpDownCore(args)

class AllImgModel(BasicModel):
	def __init__(self, args):
		super(AllImgModel, self).__init__(args)
		self.core = AllImgCore(args)
