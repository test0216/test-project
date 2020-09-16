import json
import pdb
import torch
import spacy
import re
import tqdm
import stop_words
import operator
import argparse
import numpy as np
import torch.nn.functional as F
#import kg_embedding
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from collections import deque, defaultdict

from eval_metrics.bleu.bleu import Bleu
from eval_metrics.meteor.meteor import Meteor
from eval_metrics.rouge.rouge import Rouge
from eval_metrics.cider.cider import Cider
from eval_metrics.spice.spice import Spice

def open_json(path):
    with open(path, "r") as f:
        return json.load(f)

def organize_ner(ner):
    new = defaultdict(list)
    for k, v in ner.items():
        value = ' '.join(k.split())
        #if value not in stopwords:
        new[v].append(value)
    return new

def fill_random(cap, ner_dict):
    assert cap != list
    filled = []
    for c in cap:
        if c.split('_')[0] in named_entities and c.isupper():
            ent = c.split('_')[0]
            if ner_dict[ent]:
                ner = np.random.choice(ner_dict[ent])
                filled.append(ner)
            else:
                filled.append(c)
        else:
            filled.append(c)
    return filled

def get_sen_vector(sen):
        vec = torch.zeros(cn_sen_emb_size)
        num = 0.0
        for s in sen:
            try:
                emb = torch.from_numpy(wv_from_text.get_vector(s))
                vec = vec + emb
                num = num + 1
            except:
                continue
        
        if num != 0:
            vec = vec / num

        return vec
        
def rank_sentences(cap, sent, method):
    # make them unicode, spacy accepts only unicode
    cap = str(cap)
    sent_temp = [str(s) for s in sent]
    # feed them to spacy to get the vectors
    if method == 'ctg':
        if opt.dataset == 'data':
            cap = nlp(cap)
            list_sent = [nlp(s) for s in sent_temp]
        elif opt.dataset == 'cn_data':
            cap = cap.split()
            cap = get_sen_vector(cap)
            list_sent = [get_sen_vector(s) for s in sent]
    '''
    else:
    	cap = kg_embedding(cap)
    	list_sent = [kg_embedding(s) for s in sent]
    '''
    if opt.dataset == 'data':
        compare = [s.similarity(cap) for s in list_sent]
        if opt.jaccard:
            jac = torch.zeros(len(sent))
            for key, values in ner_dict.items():
                for word in values:
                    for i, s in enumerate(sent):
                        beg = s.find(word)
                        if beg is not -1:
                            jac[i] += 1
            jac = F.softmax(jac, dim=0)
            compare = torch.Tensor(compare) + jac
        similarity = sorted([(s, c) for s, c in zip(list_sent, compare)], key=lambda x: x[1], reverse=True)
    elif opt.dataset == 'cn_data':
        compare = [torch.cosine_similarity(s, cap, dim = 0) for s in list_sent]
        if opt.jaccard:
            jac = torch.zeros(len(sent))
            for key, values in ner_dict.items():
                for word in values:
                    for i, s in enumerate(sent):
                        beg = ' '.join(s).find(word)
                        if beg is not -1:
                            jac[i] += 1
            jac = F.softmax(jac, dim=0)
            compare = torch.Tensor(compare) + jac
        similarity = sorted([(s, c) for s, c in zip(sent, compare)], key=lambda x: x[1], reverse=True)

    return similarity

def ner_finder(ranked_sen, score_sen, word):
    for sen, sc in zip(ranked_sen, score_sen):
        beg = sen.find(word)
        if beg is not -1:
            end = beg + len(word)
            return sen[beg:end], sc
    else:
        return None, None

def fill_sen_emb(cap, ner_dict, ner_articles, method, return_ners=False):
    assert cap != list
    filled = []
    similarity = rank_sentences(' '.join(cap), ner_articles, method)
    if opt.dataset == 'data':
        ranked_sen = [s[0].text for s in similarity]
    elif opt.dataset == 'cn_data':
        ranked_sen = [s[0] for s in similarity]
    score_sen = [s[1] for s in similarity]
    if return_ners: ners = []

    if opt.jaccard:


    new = {}
    for key, values in ner_dict.items():
        temp = {}
        for word in values:
            found, sc1 = ner_finder(ranked_sen, score_sen, re.sub('[^A-Za-z0-9]+', ' ', word))
            found2, sc2 = ner_finder(ranked_sen, score_sen, word)
            if found:
                temp[word] = sc1
            elif ner_finder(ranked_sen, score_sen, word):
                temp[word] = sc2
            else:
                temp[word] = 0
        new[key] = temp

    try:
        new = {k: deque([i for i, _ in sorted(v.items(), key=operator.itemgetter(1), reverse=True)]) for k, v in new.items()}
    except:
        pdb.set_trace()
    for c in cap:
        if c.split('_')[0] in named_entities and c.isupper():
            ent = c.split('_')[0]
            if ner_dict[ent]:
                ner = new[ent].popleft()
                # append it again, we might need to reuse some entites.
                new[ent].append(ner)
                filled.append(ner)
                if return_ners: ners.append((ner, ent))
            else:
                filled.append(c)
        else:
            filled.append(c)
    if return_ners:
        return filled, ners
    else:
        return filled

def insert_word(ner_test, sen_att, ix, ner_dict, return_ner=False):
    if ner_test in named_entities:
        for ii in sen_att[ix]:
            if ii < len(article['sentence']):
                art_sen = article['sentence'][ii]
                temp = [(art_sen.find(ner), ner) for ner in ner_dict[ner_test] if art_sen.find(ner) != -1]
                temp = sorted(temp, key=lambda x: x[0])
                if temp and return_ner: return temp[0][1], ner_test
                if temp: return temp[0][1], None
        else:
            return ner_test, None
    else:
        return ner_test, None

def insert(cap, sen_att, ner_dict, return_ners=False):
    new_sen = ''
    words = []
    if return_ners: ners = []

    for ix, c in enumerate(cap):
        ner_test = c.split('_')[0]
        word, ner = insert_word(ner_test, sen_att, ix, ner_dict, return_ners)
        if ner:
            ners.append((word, ner))
        words.append(word)
    #         new_sen += ' ' +
    if return_ners:
        return ' '.join(words), ners
    else:
        return ' '.join(words)

def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "Spice")
    ]
    final_scores = {}
    all_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)

        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
            for m, s in zip(method, scores):
                all_scores[m] = s
        else:
            final_scores[method] = score
            all_scores[method] = scores

    return final_scores, all_scores

def evaluate(ref, cand, get_scores=True):
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    truth = {}
    for i, caption in enumerate(ref):
        truth[i] = [caption]

    # compute bleu score
    final_scores = score(truth, hypo)

    #     print out scores
    print('Bleu_1:\t ;', final_scores[0]['Bleu_1'])
    print('Bleu_2:\t ;', final_scores[0]['Bleu_2'])
    print('Bleu_3:\t ;', final_scores[0]['Bleu_3'])
    print('Bleu_4:\t ;', final_scores[0]['Bleu_4'])
    print('METEOR:\t ;', final_scores[0]['METEOR'])
    print('ROUGE_L: ;', final_scores[0]['ROUGE_L'])
    print('CIDEr:\t ;', final_scores[0]['CIDEr'])
    print('Spice:\t', final_scores[0]['Spice'])

    if get_scores:
        return final_scores
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='cn_data', help='path of dataset: data | cn_data')
    parser.add_argument('--cn_sen_emb_size', type=int, default=200, help='dimension of Chinese sentence embedding')
    parser.add_argument('--tencent_dict', type=str, default='../cn_embedding/tencent_word_vectors.bin', help='pretrained tencent embedding path')
    parser.add_argument('--output', type=str, default='./result/goodnews_output.json',
                        help='model test generation')
    parser.add_argument('--insertion_method',type=list, default=['rand', 'ctg'],
                        help='rand: random insertion, ctg: context/word2vec/glove insertion, ctk: context/word2vec/k-bert insertion, sen_att: sentence attention insertion, word_att:word attention insertion')
    parser.add_argument('--dump', type=bool, default=True,
                        help='Save the inserted captions in a json file')

    opt = parser.parse_args()
    print(opt)
    #stopwords = stop_words.get_stop_words('en')
    named_entities = ['PERSON', 'LOC', 'DATE', 'ORG', 'PRODUCT']
    #named_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
    
    fname = opt.tencent_dict
    wv_from_text = KeyedVectors.load(fname)
    cn_sen_emb_size = opt.cn_sen_emb_size
    # Start the insertion process
    output = open_json(opt.output)
    ref = []

    for h in tqdm.tqdm(output):
        ref.append(h['GT_cap'])

    for method in opt.insertion_method:
        hypo = []

        for h in tqdm.tqdm(output):
            cap = word_tokenize(h['GEN_cap'])
            if opt.dataset == 'data':
                ner_articles = h['art'].split('_-_')
            elif opt.dataset == 'cn_data':
                ner_articles = []
                for item in h['art'].split('|')[1:]:
                    ner_articles.append(item.split('_-_'))

            ner = dict(eval(h['ner']))
            ner_dict = {}
            for k in ner.keys():
                if k in h['art']:
                    ner_dict[k] = ner[k]
            
            ner_dict = organize_ner(ner_dict)

            # fill the caption with named entities
            if method == 'rand':
                cap = fill_random(cap, ner_dict)
                cap = ' '.join(cap)
                hypo.append(cap)

            elif method == 'ctg':
                cap = fill_sen_emb(cap, ner_dict, ner_articles, method)
                cap = ' '.join(cap)
                hypo.append(cap)
            '''
            elif method == 'ctk':
                cap = fill_sen_emb(cap, ner_dict, ner_articles, method)
                cap = ' '.join(cap)
                hypo.append(' '.join(cap.split()))

            elif method == 'sen_att':
                sen_att = np.array(h['sen_att']).squeeze(axis=2)
                sorted_sen_att = [s.argsort()[-55:][::-1] for s in sen_att]

                sen, name = insert(cap, sorted_sen_att, ner_dict, True)
                hypo.append(sen)

            elif method == 'word_att':
                word_att = np.array(h['word_att']).squeeze(axis=2)
                sorted_word_att = [s.argsort()[-args.word_num:][::-1] for s in word_att]

                sen, name = insert(cap, sorted_word_att, ner_dict, True)
                hypo.append(sen)
            '''
        # retrieve the reference sentences
        if opt.dump:
            json.dump(hypo, open('./result/%s.json' % method, 'w'))
        print('Insertion Method: %s' % method)
        sc, scs = evaluate(ref, hypo)###

