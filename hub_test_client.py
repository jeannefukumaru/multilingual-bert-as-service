import zmq 
import time 
import numpy
import jsonlines
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
    with jsonlines.open(jsonl) as reader: 
        msg_gen = (obj for obj in reader)
    return msg_gen


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("jsonl", help='path to jsonl file containing new line delimited lines of text for encoding')
    args = parser.parse_args()

    msg_gen = data_gen(args.jsonl)
    # send data to server 
    for i, m in enumerate(msg_gen):
        print("sending request %s" % i)
        start = time.time()
        socket.send_json(m)
        messages = recv_array(socket)
        end = time.time()
        time_taken = end-start
        print(f"received reply {messages} processing took {time_taken}")
