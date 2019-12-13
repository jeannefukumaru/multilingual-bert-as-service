import sys
import time 
import zmq
from transformers import BertModel, BertTokenizer
from utils import send_array_and_str, recv_array_and_str, preprocess
import torch
from multiprocessing import Process


def run_worker():
    # set up zmq
    context = zmq.Context()
    # socket to receive work from ventilator
    work_receiver = context.socket(zmq.PULL).connect('tcp://localhost:5557')
    # socket to send results to sink
    results_sender = context.socket(zmq.PUSH).connect('tcp://localhost:5558')
    print('bound sockets')
    # set up model 
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    print('loaded pre-trained model')
    model.eval()
    while True:
        sentence, tokens_tensor = recv_array_and_str(work_receiver)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            encoded_layers = outputs[0].numpy()
            send_array_and_str(results_sender, encoded_layers, sentence)
            print('sent response')

