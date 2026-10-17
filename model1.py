# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 20:10:20 2023

@author: HP
"""
import simpletransformers
from simpletransformers.ner import NERModel, NERArgs
import torch


class MyNERModel:
    def __init__(self, model_type, model_name, labels):
        self.args = NERArgs()
        self.args.train_batch_size = 32
        self.args.evaluate_during_training = True
        self.args.num_train_epochs = 10
        self.model = NERModel(model_type, model_name, use_cuda=False, labels=labels, args=self.args)
        
    def load(self, model_path):
        if torch.cuda.is_available():
            self.model = torch.load(model_path)
        else:
            self.model = torch.load(model_path, map_location=torch.device('cpu') )
            self.model.device = torch.device('cpu')

#    def train(self, train_file, val_file):
#        self.model.train_model(train_file, eval_data=val_file, args={"overwrite_output_dir": True})

    def predict(self, data_file):
        preds_list = self.model.predict(data_file)
        return preds_list

#    def save(self, model_path):
#        torch.save(self.model, model_path)

#    def evaluate(self, sentence):
#        result, model_outputs, preds_list = self.model.predict([sentence])
#        return result
