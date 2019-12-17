# for creating multilingual emb viz 
datapath = 'benchmark/data/processed/by_language.csv'
df = pd.read_csv(datapath)

# sentence-level encodings
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

# word-level encodings 
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