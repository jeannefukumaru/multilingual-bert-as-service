from absl import logging

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import zmq
import torch
from transformers import BertTokenizer, BertModel
from utils import send_array, recv_array
from argparse import ArgumentParser

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.eval()

def preprocess(text, tokenizer):
    '''tokenize text into subwords and convert to indices
    :param text str: text to be preprocessed 
    :param tokenizer: BertTokenizer object
    :output: torch tensor of vocab ids
    '''
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor

if __name__=="__main__":
    parser = ArgumentParser('setup bert-multilingual for serving')
    parser.add_argument("model-dir", help="model where pretrained model is stored")
    args = parser.parse_args()
    while True:  
        message = socket.recv_json()
        print('received request: %s' % message)
        # add generator prefetch option
        tokens_tensor = preprocess(message, tokenizer)
        # Predict hidden states features for each layer
        with torch.no_grad():
            # See the models docstrings for the detail of the inputs
            outputs = model(tokens_tensor)
            # Transformers models always output tuples.
            # See the models docstrings for the detail of all the outputs
            # In our case, the first element is the hidden state of the last layer of the Bert model
            encoded_layers = outputs[0].numpy()
            send_array(socket, encoded_layers)

