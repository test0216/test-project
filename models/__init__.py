import torch
from .vonly_model import *
from .sen_emb_model import *

#from .sen_model import *

def setup(args):
    
    # show and tell model
    if args.caption_model == 'show_tell':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = ShowTellModel(args)
        params = list(model.core.parameters())
        
    # show, attend and tell model
    elif args.caption_model == 'show_attend_tell':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = ShowAttendTellModel(args)
        params = list(model.core.parameters())

    # img is concatenated with word embedding at every time step as the input of lstm
    elif args.caption_model == 'all_img':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = AllImgModel(args)
        params = list(model.core.parameters())

    # Att2in model with two-layer MLP img embedding and word embedding
    elif args.caption_model == 'att2in2':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = Att2in2Model(args)
        params = list(model.core.parameters())

    # Adaptive Attention model from Knowing when to look
    elif args.caption_model == 'adaatt':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = AdaptAttendModel(args)
        params = list(model.core.parameters())

    # Top-down attention model
    elif args.caption_model == 'updown':
        assert args.sen_emb == False and args.use_word == False, 'Don\'t use article embedding'
        model = UpDownModel(args)
        params = list(model.core.parameters())
    
    # Multi-scale attention model
    elif args.caption_model == 'msatt':
        assert args.sen_emb == True and args.use_word == True, 'MSAtt should use sentence and word embedding'
        model = Encoder2Decoder(args)
        params = list(model.embed_vocab.parameters()) + list(model.GRU.parameters()) + list(model.mlp.parameters())

        if args.first_only:
            params = params + list(model.atten_1st.parameters())
        elif args.second_only:
            params = params + list(model.atten_2nd.parameters())
        else:
            params = params + list(model.atten_1st.parameters()) + list(model.atten_2nd.parameters())

    # Show attend and tell model with article embedding
    elif args.caption_model == 'goodnews':
        assert args.sen_emb == True and args.use_word == False, 'GoodNews should only use sentence embedding'
        model = Encoder2Decoder(args)
        params = list(model.embed_vocab.parameters()) + list(model.atten_1st.parameters()) \
               + list(model.GRU.parameters()) + list(model.mlp.parameters())
    
    else:
        raise Exception("Caption model not supported: {}".format(args.caption_model))

    # check compatibility if training is continued from previously saved model
    if args.reload:
        start_epoch = int(args.reload_model_path.split('_')[1]) + 1
        model.load_state_dict(torch.load(args.reload_model_path))
        print('model load from: {}'.format(args.reload_model_path))
        
        return model, params, start_epoch
        
    return model, params