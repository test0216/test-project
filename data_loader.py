import os
import pdb
import spacy
import json
import torch
import string
import numpy as np
from gensim.models import KeyedVectors
from itertools import groupby
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

class Dataset(data.Dataset):
    
    def __init__(self, args, transform=None):
        self.dataset = args.dataset
        self.fname = args.tencent_dict
        self.lower = args.lower 
        self.sen_emb = args.sen_emb
        self.use_word = args.use_word
        self.del_stop_word = args.del_stop_word
        self.image_num = args.image_num
        self.eng_sen_emb_size = args.eng_sen_emb_size
        self.cn_sen_emb_size = args.cn_sen_emb_size

        if self.dataset == 'data':
            self.max_caption_len = 31
            self.word_num = args.word_num
            self.image_dir = args.image_dir
        elif self.dataset == 'cn_data':
            self.max_caption_len = 15
            self.word_num = args.cn_word_num
            self.image_dir = args.cn_image_dir
            self.wv_from_text = KeyedVectors.load(self.fname)

        self.text_train_path = os.path.join('..', args.dataset, args.text_train_path)

        if self.lower:
            self.vocab_path = os.path.join('..', args.dataset, args.vocab_wtoi_lower_path)
            self.vocab_size = args.vocab_lower_size
        else:
            self.vocab_path = os.path.join('..', args.dataset, args.vocab_wtoi_path)
            if self.dataset == 'data':
                self.vocab_size = args.vocab_size
            elif self.dataset =='cn_data':
                self.vocab_size = args.cn_vocab_size

        self.transform = transform

        with open(self.text_train_path, 'r') as f:
            self.text = json.load(f)
        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f) 

        self.ids = len(self.text)
        self.nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger'])
        self.nlp_word = spacy.load('en', disable=['parser', 'tagger'])

    def __getitem__(self, index):
        data = self.text[index]   
        
        if self.sen_emb:
            if self.del_stop_word:
                article = data['art_without_sw']
            else:
                if self.dataset == 'data':
                    article = data['article']
                else:
                    article = data['article_comma_seg']
                    ner = data['ner']

            if self.dataset == 'data':
                #article
                sen_len = 54
                temp = np.zeros([sen_len + 1, self.eng_sen_emb_size])

                if len(article) < sen_len + 1:
                    for i, sen in enumerate(article):
                        temp[i, :] = self.get_word_vector(sen.lower().translate(str.maketrans('', '', string.punctuation)).strip())
                else:
                    for i, sen in enumerate(article[:sen_len]):
                        temp[i, :] = self.get_word_vector(sen.lower().translate(str.maketrans('', '', string.punctuation)).strip())
                    temp[sen_len, :] = np.average([self.get_word_vector(sen.lower()) for sen in article[sen_len:]]) 
                article_emb = temp.tolist()
            
                if self.use_word:
                    #words
                    words = self.emb_loop(article)
                    word_id = words.tolist()
            elif self.dataset == 'cn_data':
                #article
                sen_len = 14
                temp = np.zeros([sen_len + 1, self.cn_sen_emb_size])

                if len(article) < sen_len + 1:
                    for i, sen in enumerate(article):
                        temp[i, :] = self.get_sen_vector(sen)
                else:
                    for i, sen in enumerate(article[:sen_len]):
                        temp[i, :] = self.get_sen_vector(sen)
                    temp[sen_len, :] = np.average([self.get_sen_vector(sen) for sen in article[sen_len:]])
                article_emb = temp.tolist()
                
                #words
                if self.use_word:
                    words, words_num = self.id_loop(article, ner, sen_len + 1)
                    word_id = words.tolist()

        # caption
        cap = data['tokens_temp']
        try:
            tokens = cap
        except:
        	print(caption)
        caption = []
        caption.append(self.vocab['<start>'])
        
        if self.lower:
            for i, token in enumerate(tokens):
                if token.endswith('_'):                   
                    try:
                        a = self.vocab[token]
                    except:
                        a = self.vocab_size
                else:
                    try:
                        a = self.vocab[token.lower()]
                    except:
                        a = self.vocab_size
                caption.append(a)
                if i == self.max_caption_len - 1:
                    break
        else:
            for i, token in enumerate(tokens):
                try:
                    a = self.vocab[token]
                except:
                    a = self.vocab_size
                caption.append(a)
                if i == self.max_caption_len - 1:
                    break

        caption.append(self.vocab['<end>'])
        caption = torch.Tensor(caption)

        # image
        img_id = data['imgid']
        if self.dataset == 'data':
        	#single image
            path = img_id + '.jpg'
            images = Image.open(os.path.join(self.image_dir, path)).convert('RGB')

            if self.transform is not None:
                images = self.transform(images)
            img_num = 1

        elif self.dataset == 'cn_data':
        	#multiple images
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
        
        if self.sen_emb:
            if self.use_word:
                return images, img_num, caption, article_emb, word_id, words_num
            else:
                return images, img_num, caption, article_emb
        else:
            return images, img_num, caption

    def __len__(self):
        return self.ids
    
    def id_loop(self, article, ner, sen_len):
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
                a = self.vocab[t]
            except:
                a = self.vocab_size
            token_id.append(a)

        return token_id 

def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=True)
    try:
        images, img_num, captions, article_emb, word_emb, words_num = zip(*data) # unzip
    except:
        try:
            images, img_num, captions, article_emb = zip(*data)
        except:
            images, img_num, captions = zip(*data)
    
    try:
        article_emb
    except NameError:
        art_exist = False
    else:
        art_exist = True

    try:
        word_emb
    except NameError:
        word_exist = False
    else:
        word_exist = True

    if art_exist:
        article = []
        for emb in article_emb:
            article.append(emb)
        article = torch.Tensor(article)
    
    if word_exist:
        word = []
        for emb in word_emb:
            word.append(emb)
        word = torch.Tensor(word).long()

        word_num = []
        for num in words_num:
            word_num.append(num)
        word_num = torch.Tensor(word_num) 
        
    lengths = [len(cap) for cap in captions]

    # Merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor)
    target = torch.zeros(len(captions), 17).long() #33 or 17
    for i, cap in enumerate(captions):
        end = lengths[i]
        target[i, :end] = cap[:end]
    
    batch = {}
    batch['images'] = images
    batch['img_num'] = img_num
    batch['captions'] = target
    batch['lengths'] = lengths
    if word_exist:
        batch['article'] = article
        batch['word'] = word
        batch['word_num'] = word_num
        return batch
    elif art_exist:
        batch['article'] = article
        return batch
    else:
        return batch

def get_loader(args):
    transform = transforms.Compose([ 
                transforms.CenterCrop(args.crop_size),
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                     (0.229, 0.224, 0.225))])
    
    data = Dataset(args, transform = transform)

    data_loader = torch.utils.data.DataLoader(dataset = data,
                                              batch_size = args.batch_size,
                                              shuffle = args.train_shuffle,
                                              num_workers = args.num_workers,
                                              collate_fn = collate_fn,
                                              pin_memory = True)
    return data_loader
