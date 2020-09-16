import argparse
import opts
import json
import torch
import models
import torch.nn as nn
from utils import eval_metrics


def main(args):
    
    args.reload = True
    model, _, _ = models.setup(args)
    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    score, output = eval_metrics(model, args)

    bleu_1 = score['Bleu_1'] 
    bleu_2 = score['Bleu_2']
    bleu_3 = score['Bleu_3']
    bleu_4 = score['Bleu_4']
    meteor = score['METEOR']
    rouge_l = score['ROUGE_L']
    cider = score['CIDEr']
    spice = score['SPICE']
    
    print('Model best score: bleu_1:%.5f, bleu_2:%.5f, bleu_3:%.5f, bleu_4:%.5f, meteor:%.5f, rouge_l:%.5f, cider:%.5f, spice:%.5f\n' \
                         % (bleu_1, bleu_2, bleu_3, bleu_4, meteor, rouge_l, cider, spice))
    
    out_name = './result/' + args.caption_model + '_output.json'
    with open(out_name, 'w') as f:
        json.dump(output, f)

if __name__ == '__main__':
    
    args = opts.parse_opt()
    print(args)
    main(args)
