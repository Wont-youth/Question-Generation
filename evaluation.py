# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
import numpy as np

from dataset import Seq2SeqDataset
from torch.utils.data import DataLoader

from model.seq2seq import Seq2Seq
from model.utils import EOS_INDEX, PAD_INDEX, sentence_clip, tokenize

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from rouge import Rouge
from model.metric import calculate_bleu, calculate_meteor, calculate_rough

import pickle

random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Tester(object):

    def __init__(self, config):
        self._config = config
    
    def make_model(self):
        model = Seq2Seq(
            vocab_size=self._config.vocab_size,
            embed_size=self._config.embed_size,
            hidden_size=self._config.hidden_size,
            rnn_type=self._config.rnn_type,
            num_layers=self._config.num_layers,
            bidirectional=self._config.bidirectional,
            attention_type=self._config.attention_type,
            dropout=self._config.dropout
        )
        model.load_pretrained_embeddings(self._config.embedding_file_name)
        return model

    def make_data(self):
        test_dataset = Seq2SeqDataset(self._config.test_path)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            pin_memory=True
        )
        return test_loader

    def make_vocab(self):
        with open(self._config.vocab_path, 'rb') as handle:
            self._index2word = pickle.load(handle)

    def test(self):
        self.make_vocab()
        test_loader = self.make_data()
        
        model = self.make_model()
        model.load_state_dict(torch.load(self._config.model_path))
        model = model.cuda()
        model.eval()
        
        pred = []
        for data in test_loader:
            src, trg = data
            trg_mask = trg != PAD_INDEX
            trg_lens = trg_mask.long().sum(dim=1, keepdim=False)
            src = src.cuda()
            with torch.no_grad(): # beam decode
                # output = model.decode(src, trg_lens.max().item() + 1)
                output = model.beam_decode(src, trg_lens.max().item() + 1, beam_size=3)
                texts = self.ndarray2texts(output.cpu().numpy())
                pred.extend(texts)
        path = './data/output/test.txt'
        self.write_file(pred, path)
        bleu_1, bleu_2, bleu_4, meteor, rough_l = self.get_metric(path, self._config.test_reference_path)
        print(f'bleu-1, bleu-2, bleu-4, meteor, rough-l: {bleu_1}, {bleu_2}, {bleu_4}, {meteor}, {rough_l}')

    def ndarray2texts(self, ndarray):
        texts = []
        for vector in ndarray:
            text = ''
            for index in vector:
                if index == EOS_INDEX or index == PAD_INDEX:
                    break
                elif self._index2word[index] == '?':
                    text += self._index2word[index] + ' '
                    break
                text += self._index2word[index] + ' '
            texts.append(text.strip())
        return texts
        
    def write_file(self, texts, path):
        file = open(path, 'w', encoding=u'utf-8')
        for text in texts:
            file.write(text + '\n')
    
    def get_metric(self, hypotheses_path, references_path):
        hypotheses = open(hypotheses_path, 'r', encoding='utf-8').readlines()
        references = open(references_path, 'r', encoding='utf-8').readlines()
        hypotheses = [x.strip() for x in hypotheses]
        references = [x.strip() for x in references]
        assert len(hypotheses)==len(references)
        print(f"The length of test set is {len(hypotheses)}. Start calculating...")
        bleu_1_list, bleu_2_list, bleu_4_list, meteor_list, rough_l_list = [], [], [], [], []
        for i in range(len(references)):
            hypothese = hypotheses[i]
            reference = references[i]
            hypothese_split = hypothese.split()
            reference_split = [reference.split()]
            bleu_1_gram, bleu_2_gram, bleu_4_gram = calculate_bleu(reference_split, hypothese_split)
            bleu_1_list.append(bleu_1_gram)
            bleu_2_list.append(bleu_2_gram)
            bleu_4_list.append(bleu_4_gram)
            meteor_score = calculate_meteor(reference_split, hypothese_split)
            meteor_list.append(meteor_score)
            rouge = Rouge()
            rough_score = rouge.get_scores(hypothese, reference)
            rough_l = rough_score[0]["rouge-l"]
            rough_l_r = rough_l["r"]
            rough_l_list.append(rough_l_r)
        bleu_1 = round(sum(bleu_1_list)*100/len(bleu_1_list), 3)
        bleu_2 = round(sum(bleu_2_list)*100/len(bleu_2_list), 3)
        bleu_4 = round(sum(bleu_4_list)*100/len(bleu_4_list), 3)
        meteor = round(sum(meteor_list)*100/len(meteor_list), 3)
        rough_l = round(sum(rough_l_list)*100/len(rough_l_list), 3)
        return bleu_1, bleu_2, bleu_4, meteor, rough_l


parser = argparse.ArgumentParser()
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--attention_type', type=str, default='ScaledDot', choices=['Dot', 'ScaledDot', 'Concat', 'Bilinear', 'MLP'])
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=43946) # vocab size, must modify after preprocessing
parser.add_argument('--hidden_size', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--embedding_file_name', type=str, default='data/processed/glove.npy')
parser.add_argument('--model_path', type=str, default='./data/checkpoints/model-epoch-7.params')
parser.add_argument('--vocab_path', type=str, default='./data/processed/index2word.pkl')
parser.add_argument('--test_path', type=str, default='./data/processed/test.npz')
parser.add_argument('--test_reference_path', type=str, default='./data/raw/trg_test.txt')
parser.add_argument('--gpu_id', type=str, default='0')

config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

Tester = Tester(config)

Tester.test()
