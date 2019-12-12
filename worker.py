import sys
import time 
import zmq
from transformers import BertModel
from utils import send_array, recv_array
import torch

class Worker(model):
    def __init__(self):
        self.model = BertModel.from_pretrained(model).eval()
    def run(self):
        # setup zmq
        context = zmq.Context()
        # socket to receive work from ventilator
        work_receiver = context.socket(zmq.PULL).connect("tcp://localhost:5557")
        # socket to send results to sink
        results_sender = context.socket(zmq.PUSH).connect("tcp://localhost:5558")
        while True:
            tokens_tensor = recv_array(work_receiver)
            with torch.no_grad():
                outputs = model(tokens_tensor)
                encoded_layers = outputs[0].numpy()
                send_array(results_sender, encoded_layers)
