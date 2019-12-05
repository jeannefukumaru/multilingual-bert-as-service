import zmq 
import time 
import numpy
from utils import send_array, recv_array

context = zmq.Context()

print('connecting to bert-multilingual server')
socket = context.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')

# Compute a representation for each message, showing various lengths supported.
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]

for i, m in enumerate(messages):
    print("sending request %s" % i)
    start = time.time()
    socket.send_json(m)
    messages = recv_array(socket)
    end = time.time()
    time_taken = end-start
    print(f"received reply {messages} processing took {time_taken}")
