import zmq 
import time 
import numpy
import json
import jsonlines
import pandas as pd 
from argparse import ArgumentParser
from utils import send_array_and_str, preprocess
from transformers import BertTokenizer 

def data_gen(jsonl):
    '''takes in jsonl file and yields generator for feeding data to requests
    :param:jsonl: path to jsonl file with text data 
    :return: python generator for passing text to client''' 
    for msg in jsonlines.open(jsonl): 
        yield msg

def client():
    # setup bert tokenizer 
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # create zmq context
    context = zmq.Context()

    ventilator = context.socket(zmq.PUSH)
    ventilator.connect('tcp://127.0.0.1:5555')
    print('bound to ventilator')

    sink = context.socket(zmq.SUB)
    sink.bind('tcp://127.0.0.1:5556')
    print('bound to sink')
    
    # read sentences, tokenize and send to server 
    sentences = ['hamburgers with jalapeno', 'pizza with pepperoni', 'cupcakes with caramel']

    for m in sentences:
        print('tokenizing sentence before sending')
        tokens_tensor = preprocess(m, tokenizer)
        send_array_and_str(ventilator, tokens_tensor, m)
        print('msg sent')

if __name__=='__main__':
    client()