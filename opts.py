import argparse

def parse_opt():

    parser = argparse.ArgumentParser() 

    #path
    parser.add_argument('--dataset', type=str, default='cn_data', help='path of dataset: data | cn_data')

    parser.add_argument('--vocab_wtoi_lower_path', type=str, default='vocab_wtoi_short.json', help='path for vocabulary wrapper') #word to id for lower vocabulary
    parser.add_argument('--vocab_itow_lower_path', type=str, default='vocab_itow_short.json', help='path for vocabulary wrapper') #id to word for lower vocabulary        

    parser.add_argument('--vocab_wtoi_path', type=str, default='vocab_wtoi.json', help='path for vocabulary wrapper') #word to id for normal vocabulary
    parser.add_argument('--vocab_itow_path', type=str, default='vocab_itow.json', help='path for vocabulary wrapper') #id to word for normal vocabulary                

    parser.add_argument('--text_train_path', type=str, default='data_train_10.json', help='path for train text json file') #train data path (default: data_train.json)
    parser.add_argument('--text_test_path', type=str, default='data_test_10.json', help='path for test text json file') #valuation data path (default: data_val.json)
    parser.add_argument('--text_val_path', type=str, default='data_test_10.json', help='path for valuate text json file') #test data path (default: data_test.json)
    
    parser.add_argument('--save_name', type=str, default='modelshowattendtell_', help='name of the saved model') #model save as this name
    parser.add_argument('--reload_model_path', type=str, default='./save/modelshowtell_32_.pth', help='path to saved model') #during training, the path for reloading model

    parser.add_argument('--image_dir', type=str, default='../../GoodNews-master/resized', help='path for images') # as help
    parser.add_argument('--cn_image_dir', type=str, default='../cn_data/Images', help='path for images') # as help
    parser.add_argument('--glove_path', type=str, default='glove_embedding.npy', help='pretrained glove embedding weight') 
    parser.add_argument('--tencent_path', type=str, default='tencent_embedding.npy', help='pretrained tencent embedding weight')
    parser.add_argument('--tencent_dict', type=str, default='../cn_embedding/tencent_word_vectors.bin', help='pretrained tencent embedding path')
    
    #tricks bool
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--schedule', type=bool, default=False) 
    parser.add_argument('--converge', type=bool, default=False)
    parser.add_argument('--del_stop_word', type=bool, default=False, help='use articles without stopwords')
    parser.add_argument('--use_trick', type=bool, default=False, help='when use trick, remember switch this to true')
    parser.add_argument('--random', type=bool, default=False, help='random sampling from the vocabulary size possibility distribution')
    parser.add_argument('--train_shuffle', type=bool, default=True, help='whether or not random select samples from the training set')
    parser.add_argument('--use_crossentropy', type=bool, default=False, help='whether or not use cross entropy as training loss')
    parser.add_argument('--return_att', type=bool, default=False, help='used when appling sen_att or word_att method in insertion part')
    parser.add_argument('--add', type=bool, default=False, help='in the last step, whether to use add or concat')
    parser.add_argument('--trans_att', type=bool, default=False, help='transport attention weight from the first attention to second attention')
    parser.add_argument('--atten_method', type=str, default='concat', help='attention method: add | cosine | concat | multiply')

    parser.add_argument('--reload', type=bool, default=False, help='Whether to load existed model')
    parser.add_argument('--sen_emb', type=bool, default=False, help='Whether to use article embedding (for GoodNews model and our model)')
    parser.add_argument('--use_word', type=bool, default=False, help='Whether to use word embedding (for our model only)')
    parser.add_argument('--lower', type=bool, default=False, help='Whether to use lower words')
    parser.add_argument('--first_only', type=bool, default=False, help='Whether to use the first attention only (only for our model')
    parser.add_argument('--second_only', type=bool, default=False, help='Whether to use the second attention only (only for our model')
    
    #run parameters
    parser.add_argument('--num_epochs', type=int, default=50) #50
    parser.add_argument('--num_workers', type=int, default=4, help='When using ibex, set to 8 or 16. Ortherwise, set to 1')
 
    parser.add_argument('--log_step', type=int, default=4, help='step size for printing log info')#10
    parser.add_argument('--start_save', type=int, default=5, help='start epoch for saving model')
    parser.add_argument('--save_step', type=int, default=5, help='step size for saving model after epoch that the save_step set')
    parser.add_argument('--val_step', type=int, default=2, help='step size for valuate model')

    parser.add_argument('--batch_size', type=int, default=16, help='512 or 256 (depend on how many GPU you use, 64 or 32 for each)')
    parser.add_argument('--eval_size', type=int, default=16, help='256 or 128 (depend on how many GPU you use, 32 or 16 for each)')

    parser.add_argument('--image_num', type=int, default=15, help='number of images')
    parser.add_argument('--word_num', type=int, default=1000, help='use how many words from the article') 
    parser.add_argument('--cn_word_num', type=int, default=65, help='use how many words from the Chinese article')###

    parser.add_argument('--vocab_size', type=int, default=35391, help='size of the vocabulary')
    parser.add_argument('--cn_vocab_size', type=int, default=34016, help='size of the chinese vocabulary')
    parser.add_argument('--vocab_lower_size', type=int, default=29720, help='size of the vocabulary when use lower words')

    parser.add_argument('--caption_model', type=str, default='show_attend_tell', \
                        help='model name: show_tell | show_attend_tell | all_img | att2in2 | adaatt | updown | msatt | goodnews')
    
    #K-BERT embedding parameters
    parser.add_argument("--pretrained_model_path", default='./kb_models/google_model.bin', type=str, help="Path of the pretrained model.")
    parser.add_argument("--kg_name", default='CnDbpedia', help="KG name or path")    
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset") 
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", "cnn", "gatedcnn", "attn", "rcnn", "crnn",           
                                              "gpt", "bilstm"], default="bert", help="Encoder type.")
                                                 
    #hyperparameters
    parser.add_argument('--drop_prob', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--cnn_epoch', type=int, default=20, help='start fine-tuning CNN after')        
    parser.add_argument('--lr_decay', type=int, default=10, help='epoch at which to start lr decay')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layer of rnn')
 
    parser.add_argument('--learning_rate_decay_every', type=int, default=50, help='decay learning rate at every this number')            
    parser.add_argument('--fine_tune_start_layer', type=int, default=5, help='CNN fine-tuning layers from: [0-7]')                
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate for the whole model')#4e-4 
    parser.add_argument('--learning_rate_cnn', type=float, default=1e-4, help='learning rate for fine-tuning CNN')  

    parser.add_argument('--clip', type=float, default=0.1) 
    parser.add_argument('--lambda1', type=float, default=0.8)
    parser.add_argument('--lambda2', type=float, default=0.2)            
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')  
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha in Adam')                    
    parser.add_argument('--beta', type=float, default=0.999, help='beta in Adam') 

    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')               
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states') 
    parser.add_argument('--img_feat_size', type=int, default=2048, help='dimension of image features')
    parser.add_argument('--eng_sen_emb_size', type=int, default=300, help='dimension of English sentence embedding')
    parser.add_argument('--cn_sen_emb_size', type=int, default=200, help='dimension of Chinese sentence embedding')

    args = parser.parse_args()

    return args