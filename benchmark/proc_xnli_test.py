import pandas as pd 
import jsonlines
import json
from config import config


with jsonlines.open(config['raw_datafile']) as reader: 
    data = [obj for obj in reader]

df = pd.DataFrame(data)

sentences1 = df['sentence1']
sentences2 = df['sentence2']

all_sentences = sentences1.append(sentences2).tolist()
assert len(all_sentences)==150300

with open(config['processed_datafile'], 'w') as outfile:
    for sent in all_sentences:
        json.dump(sent, outfile)
        outfile.write('\n')