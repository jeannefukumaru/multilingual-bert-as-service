import zmq 
import time 
import numpy
import json
import jsonlines
import pandas as pd 
from argparse import ArgumentParser
from utils import send_array, recv_array, preprocess
from transformers import BertTokenizer 

# setup bert tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# # create zmq context
# context = zmq.Context()

# print('connecting to bert-multilingual server')
# ventilator = context.socket(zmq.PUSH)
# ventilator.connect('tcp://localhost:5555')

# print('connecting to sink')
# sink = context.socket(zmq.SUB)
# sink.connect('tcp://localhost:5556')

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
    # read sentences, tokenize and send to server 
    df = pd.DataFrame(columns=['sentence','embedding'])
    for i, m in enumerate(msg_gen):
        print('tokenizing sentence before sending')
        tokens_tensor = preprocess(m, tokenizer)
        print(f"sending job {i}")
        ventilator.send_array(ventilator,m)
        message = recv_array(sink)
        print(f"received reply {message}")
        msg_dictionary = {'sentence':m, 'embedding':message}
        df = df.append(msg_dictionary, ignore_index=True)
        print('reply written to file')
    end = time.time()
    time_taken = end-start
    df.to_csv('processed_xnli.csv', index=False)
    print(f"processing took {time_taken}")

# for creating multilingual emb viz 
df = pd.read_csv('by_language.csv')

def get_encs(col):
    with torch.no_grad():
        for s in col:
            tokens = preprocess(s, tokenizer)
            torch_tokens = torch.from_numpy(tokens)
            output = model(torch_tokens)
            enc = output[0].numpy()
            return enc

df['sent1_encs'] = df['sentence1'].apply(get_encs)
df['sent2_encs'] = df['sentence2'].apply(get_encs)
df.to_json('by_lang_with_embeddings.json')