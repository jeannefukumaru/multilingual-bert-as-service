import zmq 
import time 
import numpy
import json
import jsonlines
import pandas as pd 
from argparse import ArgumentParser
from utils import send_array, recv_array

# create zmq context
context = zmq.Context()

print('connecting to bert-multilingual server')
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')

def data_gen(jsonl):
    '''takes in jsonl file and yields generator for feeding data to requests
    :param:jsonl: path to jsonl file with text data 
    :return: python generator for passing text to client''' 
    for msg in jsonlines.open(jsonl): 
        yield msg

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonl", help='path to jsonl file containing new line delimited lines of text for encoding')
    args = parser.parse_args()
    start = time.time()
    msg_gen = data_gen(args.jsonl)
    # send data to server 
    df = pd.DataFrame(columns=['sentence','embedding'])
    for i, m in enumerate(msg_gen):
        print("sending request %s" % i)
        socket.send_json(m)
        message = recv_array(socket)
        print(f"received reply {message}")
        msg_dictionary = {'sentence':m, 'embedding':message}
        df.append(msg_dictionary, ignore_index=True)
        print('reply written to file')
    end = time.time()
    time_taken = end-start
    df.to_csv('processed_xnli.csv', index=False)
    print(f"processing took {time_taken}")