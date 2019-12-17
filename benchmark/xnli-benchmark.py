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

def get_encs(sentence):
    with torch.no_grad():
        print(sentence)
        tokenized_text = tokenizer.tokenize(sentence)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        print(len(indexed_tokens))
        indexed_tokens = torch.tensor(indexed_tokens).unsqueeze(0)
        output = model(indexed_tokens)
        enc = output[0].numpy()
        print(enc.shape)
        return enc

df['sent1_encs'] = df['sentence1'].apply(get_encs)
df['sent2_encs'] = df['sentence2'].apply(get_encs)
df.to_pickle('by_lang_with_embeddings.pkl')
lang = pd.read_pickle('by_lang_with_embeddings.pkl')

# get individual word embeddings from sentences 
lang['encs1'] = [l.squeeze(0) for l in lang['sent1_encs']]
encs1_concat = np.concatenate(encs1)
tok1 = df['sentence1'].apply(tokenize)
tok1_ls = 
encs2 = [l.squeeze(0) for l in lang['sent2_encs']]
encs2_concat = np.concatenate(encs2)

# vocab 
df = pd.read_csv('by_language.csv')

def tokenize(sentence):
    with torch.no_grad():
        tokenized_text = tokenizer.tokenize(sentence)
        return tokenized_text

tok1 = df['sentence1'].apply(tokenize)
tok2 = df['sentence2'].apply(tokenize)
tok_ls1 = list(set([t for tok in tok1 for t in tok]))
print(len(tok_ls1))
tok_ls2 = list(set([t for tok in tok2 for t in tok]))
print(len(tok_ls2))
tok_ls = tok_ls1 + tok_ls2
assert len(tok_ls) == len(tok_ls1) + len(tok_ls2)
idx = tokenizer.convert_tokens_to_ids(tok_ls)
assert len(tok_ls) == len(idx)
stoi = {k:v for k in tok_ls for v in idx}
vec_df = pd.DataFrame(columns=['word','id'])

def get_encs2(id):
    with torch.no_grad():
        torch_tensor = torch.tensor(np.array(id).reshape([1,1]))
        output = model(torch_tensor)
        enc = output[0].numpy().squeeze(0)
        return enc
vec_df['word'] = tok_ls
vec_df['id'] = idx
vec_df['embedding'] = vec_df['id'].apply(get_encs2)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('run')
words = vec_df['word'].to_list()
embs = np.concatenate(vec_df['embedding'])
writer.add_embedding(embs, metadata=words)

def nn():