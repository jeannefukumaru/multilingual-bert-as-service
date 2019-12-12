import zmq 
import random 
import time 
import numpy as np 
import time 
import torch 
from utils import send_array, recv_array, preprocess
from argparse import ArgumentParser 

class Ventilator(): 
    def __init__(self):
        # zmq setup 
        self.context = zmq.Context()
        # bind ventilator to client to accept sentence strings 
        self.client = context.socket(zmq.PULL).bind("tcp://*:5555")
        # bind ventilator to worker to allocate jobs
        self.sender = context.socket(zmq.PUSH).bind("tcp://*:5557")
        # bind ventilator to sink
        self.sink = context.socket(zmq.PUSH).bind("tcp://localhost:5558")

    def run(self):
        # give everything a second to spin up and connect
        time.sleep(1)
        while True:
            tokens_tensor = recv_array(client)
            print("received request")
            send_array(sender, tokens_tensor)
            print("sent request")
            time.sleep(1)

if __name__=="__main__":
    vent = Ventilator()
    vent.run()