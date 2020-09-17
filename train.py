import json
import argparse
import torch
import os
import math
import pdb
import time
import opts
import models
import datetime
import torch.nn as nn
from utils import eval_metrics
from data_loader import get_loader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

def to_var(x):
    return Variable(x)

def main(args):
    
    if args.lower:
        with open(os.path.join('..', args.dataset, args.vocab_wtoi_lower_path), 'r') as f:
            vocab = json.load(f)
    else:
        with open(os.path.join('..', args.dataset, args.vocab_wtoi_path), 'r') as f:
            vocab = json.load(f)

    data_loader = get_loader(args) #get data loader
    
    if args.reload:
        model, params, start_epoch = models.setup(args) #get model
    else:
    	model, params = models.setup(args)
    	start_epoch = 1

    cnn_subs = list(model.cnn.resnet_conv.children())[args.fine_tune_start_layer:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]
    cnn_optimizer = torch.optim.Adam(cnn_params, lr=args.learning_rate_cnn, betas=(args.alpha, args.beta))
    '''
    for group in cnn_optimizer.param_groups:
        for param in group['params']:
            if hasattr(param, 'grad'):
                param.grad.data.clamp_(-args.clip, args.clip)
    '''
    learning_rate = args.learning_rate

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    total_step = len(data_loader)

    best_bleu_1 = 0.0
    best_bleu_2 = 0.0
    best_bleu_3 = 0.0
    best_bleu_4 = 0.0
    best_meteor = 0.0
    best_rouge_l = 0.0
    best_cider = 0.0
    best_spice = 0.0

    # Start Training
    for epoch in range(start_epoch, args.num_epochs + 1):    

        # Start Learning Rate Decay
        if epoch > args.lr_decay and (epoch - 10) % 8 == 0:
                
            #frac = float(epoch - args.lr_decay) / args.learning_rate_decay_every
            #decay_factor = math.pow(0.5, frac)

            # Decay the learning rate
            learning_rate = args.learning_rate * 0.8
        
        print('Learning Rate for Epoch %d: %.6f' % (epoch, learning_rate))

        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(args.alpha, args.beta))
        
        print('------------------Training for Epoch %d----------------'%(epoch))  
        
        for i, batch in enumerate(data_loader):

            #time_stamp = datetime.datetime.now()
            #print(time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))

            images = batch['images']
            img_num = batch['img_num']
            captions = batch['captions']
            lengths = batch['lengths']
            
            if args.sen_emb:
                articles = batch['article']
            if args.use_word:
                words = batch['word']
                words_num = batch['words_num']

            images = Variable(images, requires_grad=False)
            captions = Variable(captions, requires_grad=False)
            lengths = torch.Tensor([cap_len - 1 for cap_len in lengths])
            img_num = torch.Tensor(img_num)
            #cuda
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                if args.sen_emb:
                    articles = articles.cuda()
                if args.use_word:
                    words = words.cuda()

            model.train()
            model.zero_grad()
            
            mask_loss = 0.0
            conv_loss = 0.0
            
            #input data to model and return loss
            train_data = {} 
            train_data['images'] = images
            train_data['img_num'] = img_num
            train_data['captions'] = captions
            train_data['lengths'] = lengths
            if args.sen_emb:
                train_data['articles'] = articles
                if args.use_word: 
                    train_data['words'] = words
                    train_data['words_num'] = words_num
                    if args.use_trick:
                        train_data['epoch'] = epoch
                        loss, trick_loss = model(train_data)
                        loss = torch.mean(loss)
                        trick_loss = torch.mean(trick_loss)
                    else:
                        loss = model(train_data)
                        loss = torch.mean(loss)
                else:
                    loss = model(train_data)
                    loss = torch.mean(loss)
            else:
                loss = model(train_data)
                loss = torch.mean(loss)

            loss.backward()

            # Gradient clipping for gradient exploding problem in LSTM
            #for p in model.module.GRU.parameters():
            #    p.data.clamp_(-args.clip, args.clip)

            optimizer.step()

            if epoch > args.cnn_epoch:
                cnn_optimizer.step()

            if i % args.log_step == 0:
                if args.use_trick:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Trick: %5.4f' % \
                      (epoch, args.num_epochs, i, total_step, loss.data, trick_loss))
                else:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % \
                      (epoch, args.num_epochs, i, total_step, loss.data))
           
        #valuation
        if epoch % args.val_step == 0:
            score = eval_metrics(model, args, val = True)

            bleu_1 = score['Bleu_1']
            bleu_2 = score['Bleu_2']
            bleu_3 = score['Bleu_3']
            bleu_4 = score['Bleu_4']
            meteor = score['METEOR']
            rouge_l = score['ROUGE_L']
            cider = score['CIDEr']
            spice = score['SPICE']
          
            if bleu_2 > best_bleu_2:
                best_bleu_2 = bleu_2
            if bleu_3 > best_bleu_3:
                best_bleu_3 = bleu_3
            if bleu_4 > best_bleu_4:
                best_bleu_4 = bleu_4
            if meteor > best_meteor:
                best_meteor = meteor
            if rouge_l > best_rouge_l:
                best_rouge_l = rouge_l
            if cider > best_cider:
                best_cider = cider   
            if spice > best_spice:
                best_spice = spice
            if bleu_1 > best_bleu_1:
                best_bleu_1 = bleu_1
                print('Best epoch: %d, Model best score: bleu_1:%.5f, bleu_2:%.5f, bleu_3:%.5f, bleu_4:%.5f, meteor:%.5f, rouge_l:%.5f, cider:%.5f, spice:%.5f\n' \
                                 % (epoch, best_bleu_1, best_bleu_2, best_bleu_3, best_bleu_4, bets_meteor, best_rouge_l, best_cider, best_spice))
                path_name = './save/' + args.save_name + str(epoch) + '_.pth'
                torch.save(model.module.state_dict(), path_name) # module
                print('{} has been saved | loss: {}'.format(path_name, loss.data)) 
        
        #model save
        if epoch % args.save_step == 0 and epoch > args.start_save:
            path_name = './save/' + args.save_name + str(epoch) + '_.pth'
            torch.save(model.module.state_dict(), path_name) # module                
            print('{} has been saved | loss: {}'.format(path_name, loss.data))
            print('Epoch: %d, Model score: bleu_1:%.5f, bleu_2:%.5f, bleu_3:%.5f, bleu_4:%.5f, meteor:.5%, rouge_l:%.5f, cider:%.5f, spice:%.5f\n' \
                  % (epoch, bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider, spice))                                

if __name__ == '__main__':
    args = opts.parse_opt()
    print(args)
    main(args)
