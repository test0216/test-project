import os
import pdb
import json
import spacy
import torch
import string
import torch.utils.data as data
import numpy as np
from PIL import Image
from itertools import groupby
from gensim.models import KeyedVectors
from eval_metrics.bleu.bleu import Bleu
from eval_metrics.meteor.meteor import Meteor
from eval_metrics.rouge.rouge import Rouge
from eval_metrics.cider.cider import Cider
from eval_metrics.spice.spice import Spice

from torch.autograd import Variable
from torchvision import transforms, datasets

def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)

class Dataset(data.Dataset):
    
    def __init__(self, args, text_val_path, transform = None):
        self.dataset = args.dataset
        self.fname = args.tencent_dict
        self.lower = args.lower
        self.sen_emb = args.sen_emb
        self.use_word = args.use_word
        self.image_num = args.image_num
        self.del_stop_word = args.del_stop_word
        self.eng_sen_emb_size = args.eng_sen_emb_size
        self.cn_sen_emb_size = args.cn_sen_emb_size

        if self.dataset == 'data':
            self.max_caption_len = 30
            self.word_num = args.word_num
            self.image_dir = args.image_dir
        elif self.dataset == 'cn_data':
            self.max_caption_len = 14
            self.word_num = args.cn_word_num
            self.image_dir = args.cn_image_dir
            self.wv_from_text = KeyedVectors.load(self.fname)

        self.text_val_path = os.path.join('..', args.dataset, text_val_path)

        if self.lower:
            self.vocab_path = os.path.join('..', args.dataset, args.vocab_itow_lower_path)
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_path = os.path.join('..', args.dataset, args.vocab_itow_path)
            if self.dataset == 'data':
                self.vocab_size = args.vocab_size
            elif self.dataset == 'cn_data':
                self.vocab_size = args.cn_vocab_size

        self.transform = transform

        with open(self.text_val_path, 'r') as f:
            self.text = json.load(f)
        with open(os.path.join('..', args.dataset, args.vocab_wtoi_path), 'r') as f:
            self.vocab_wtoi = json.load(f)

        self.ids = len(self.text)
        self.nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])
        self.nlp_word = spacy.load('en', disable=['parser', 'tagger'])

    def __getitem__(self, index):
        
        data = self.text[index]
        
        if self.sen_emb:
            #article
            if self.del_stop_word:
                article = data['art_without_sw']
            else:
                if self.dataset == 'data':
                    article = data['article']
                elif self.dataset == 'cn_data':
                    article = data['article_comma_seg']
                    ner = data['ner']

            if self.dataset == 'data': 
                #article       
                sen_len = 54
                article_emb = np.zeros([sen_len + 1, self.eng_sen_emb_size])
                if len(article) < sen_len + 1:
                    for i, sen in enumerate(article):
                        article_emb[i, :] = self.get_word_vector(sen.lower().translate(str.maketrans('', '', string.punctuation)).strip())
                else:
                    for i, sen in enumerate(article[:sen_len]):
                        article_emb[i, :] = self.get_word_vector(sen.lower().translate(str.maketrans('', '', string.punctuation)).strip())
                    article_emb[sen_len, :] = np.average([self.get_word_vector(sen.lower()) for sen in article[sen_len:]])  
            
                if self.use_word:
                    #words
                    word_emb = self.emb_loop(article)
            
            elif self.dataset == 'cn_data':
                #article
                sen_len = 14
                article_emb = np.zeros([sen_len + 1, self.cn_sen_emb_size])

                if len(article) < sen_len + 1:
                    for i, sen in enumerate(article):
                        article_emb[i, :] = self.get_sen_vector(sen)
                else:
                    for i, sen in enumerate(article[:sen_len]):
                        article_emb[i, :] = self.get_sen_vector(sen)
                    article_emb[sen_len, :] = np.average([self.get_sen_vector(sen) for sen in article[sen_len:]])
                
                if self.use_word:
                    word_emb, words_num = self.id_loop(article, ner, sen_len + 1)
        #caption
        caption = data['tokens_temp']
        
        #image
        img_id = data['imgid']
        if self.dataset == 'data':
            #single image
            path = img_id + '.jpg'
            images = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
            if self.transform is not None:
                images = self.transform(images)
        elif self.dataset == 'cn_data':
            #multiple image
            images = torch.zeros((self.image_num, 3, 224, 224))
            for i, img in enumerate(img_id):
                path = img + '.jpg'
                image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')

                if self.transform is not None:
                    image = self.transform(image)
                
                images[i] = image

                if i + 1 == self.image_num:
                    break
            img_num = i + 1

        batch = {}
        batch['images'] = images
        batch['img_num'] = img_num
        batch['caption'] = caption
        if self.dataset == 'data':
            batch['image_ids'] = img_id
            batch['art'] = '_-_'.join(data['article'])
        elif self.dataset == 'cn_data':
            batch['image_ids'] = img_id[0]
            art = ''
            for sen in data['article_comma_seg']:
                sentence = '_-_'.join(sen)
                art = art + '|' + sentence
            batch['art'] = art
        batch['ner'] = str(data['ner'])

        if self.sen_emb:
            batch['articles'] = article_emb
            if self.use_word:
                batch['words'] = word_emb
                batch['words_num'] = words_num
                return batch
            else:
                return batch
        else:
            return batch

    def __len__(self):
        return self.ids

    def id_loop(self, article, ner):
        words = torch.zeros(self.word_num).long()
        words_num = torch.zeros(sen_len)
        word_index = 0
        for i, sentence in enumerate(article):
            for j, word in enumerate(sentence):
                if word in ner:
                    t = ner[word]
                else:
                    t = word
                try:
                    words[word_index] = self.vocab[t]
                except:
                    words[word_index] = self.vocab_size
                word_index += 1
                if word_index == self.word_num:
                    words_num[i] = j + 1
                    return words, words_num
            words_num[i] = j + 1
            if i == sen_len - 1:
                return words, words_num
        return words, words_num

    def emb_loop(self, article):
        words = torch.zeros(self.word_num).long()
        word_index = 0
        for sentence in article:
            token = sentence.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            token_id = self.get_word_id(token)
            for t in token_id:
                words[word_index] = t
                word_index += 1
                if word_index == self.word_num:
                    return words
        return words
        
    def get_word_vector(self, sen):
        sen = self.nlp(str(sen))
        return sen.doc.vector

    def get_sen_vector(self, sen):
        vec = np.zeros(self.cn_sen_emb_size)
        num = 0.0
        for s in sen:
            try:
                emb = self.wv_from_text.get_vector(s)
                vec = vec + emb
                num = num + 1
            except:
                continue
        if num != 0:
            vec = vec / num
        return vec

    def get_word_id(self, sen):
        doc = self.nlp_word(sen)
        token_id = []

        temp = [d.ent_type_+'_' if d.ent_iob_ != 'O' else d.text for d in doc]
        temp = [x[0] for x in groupby(temp)]
        for t in temp:
            try:
                a = self.vocab_wtoi[t]
            except:
                a = self.vocab_size
            token_id.append(a)

        return token_id 

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def eval_metrics(model, args, val = False):
    model.eval()

    if args.lower:
        with open(os.path.join('..', args.dataset, args.vocab_itow_lower_path), 'r') as f:
            vocab = json.load(f)
    else:
        with open(os.path.join('..', args.dataset, args.vocab_itow_path), 'r') as f:
            vocab = json.load(f)

    transform = transforms.Compose([ 
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    if val:
        eval_data_loader = torch.utils.data.DataLoader(
            dataset = Dataset(args, os.path.join('..', args.dataset, args.text_val_path), transform),
            batch_size = args.eval_size,
            shuffle = False, num_workers = args.num_workers,
            drop_last = False)
    else:
        eval_data_loader = torch.utils.data.DataLoader(
        	dataset = Dataset(args, os.path.join('..', args.dataset, args.text_test_path), transform),
        	batch_size = args.eval_size,
        	shuffle = False, num_workers = args.num_workers,
        	drop_last = False)

    results = []
    output = []
    ref = {}
    hypo = {}
    #dataset = EvalLoader(args.image_dir, args.text_val_path, vocab_size, transform)#
    for i, batch in enumerate(eval_data_loader):#
        #print('data is ready!')
        images = batch['images']
        img_num = batch['img_num']
        caption = batch['caption']
        image_ids = batch['image_ids']

        #for insertion
        art = batch['art']
        ner = batch['ner']

        images = to_var(images)

        if torch.cuda.is_available():
            images = images.cuda()
            if args.sen_emb:
                articles = torch.Tensor(batch['articles'].float()).cuda()
            if args.use_word:
                words = batch['words'].cuda()
                words_num = batch['words_num'].cuda()
        
        val_data = {}
        val_data['images'] = images
        val_data['img_num'] = img_num
        if args.sen_emb:
            val_data['articles'] = articles
            if args.use_word:
                val_data['words'] = words
                val_data['words_num'] = words_num
                generated_captions = model.module.sampler(val_data) # module
            else:
                generated_captions = model.module.sampler(val_data) # module
        else:
            generated_captions = model.module.sampler(val_data) # module

        #print('sample is ready!')
        if args.return_att:
            if args.use_word:
                gen_caption, sen_att, word_att = generated_captions.data.cpu().numpy()
            else:
                gen_caption, sen_att = generated_captions.data.cpu().numpy()
        else:
            gen_caption = generated_captions.data.cpu().numpy()

        for image_idx in range(gen_caption.shape[0]):
            sampled_ids = gen_caption[image_idx]
            sampled_caption = []

            for word_id in sampled_ids:
                try:
                    word = vocab[str(word_id)]
                except:    
                    pdb.set_trace()
                if word == '<end>':
                    break
                else:
                    sampled_caption.append(word)
            sentence = ' '.join(sampled_caption) 
            
            temp = {'image_id': image_ids[image_idx], 'caption_generate': sentence, 'caption_GT':caption[image_idx]}
            results.append(temp)

            if not val:
                out = {}
                out['art'] = art[image_idx]
                out['GEN_cap'] = sentence 
                out['GT_cap'] = caption[image_idx]
                out['ner'] = ner[image_idx]
                if args.return_att:
                    out['sen_att'] = sen_att[image_idx]
                    if args.use_word:
                        out['word_att'] = word_att[image_idx]
                output.append(out)

    try:
        print(results[:10])
    except:
        pass
    
    for item in results:
        gts = {}
        temp = []
        temp.append(item['caption_GT'])
        gts[item['image_id']] = temp
        ref.update(gts)

        res = {}
        temp = []
        temp.append(item['caption_generate'])
        res[item['image_id']] = temp
        hypo.update(res)

    score_map = score(ref, hypo)
    
    if not val:
        return score_map, output
    else:
        return score_map


