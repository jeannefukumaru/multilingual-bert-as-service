import sys
import time 
import zmq
from transformers import BertModel, BertTokenizer
from utils import send_array_and_str, recv_array_and_str, preprocess, set_logger
import torch
from multiprocessing import Process
import multiprocessing


class BertWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, graph_config):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), args.verbose)
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.is_ready = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.is_ready.clear()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def get_estimator(self):
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        model.eval()
        return model

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink_embed, *receivers):
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink_embed.connect(self.sink_address)
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
        poller = zmq.Poller()
        model = self.get_estimator()
        for sock in receivers:
            poller.register(sock, zmq.POLLIN)
        logger.info('model instantiated')
        logger.info('ready and listening!')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            events = dict(poller.poll())
            for sock_idx, sock in enumerate(receivers):
                if sock in events:
                    sentence, tokens_tensor = recv_array_and_str(sock)
                    torch_tensor = torch.from_numpy(tokens_tensor)
                    with torch.no_grad():
                        outputs = model(torch_tensor)
                        encoded_layers = outputs[0].numpy()
                        send_array_and_str(sink_embed, encoded_layers, sentence)
                        print('sent response')
                        logger.info('job done\tsize: %s\t' %encoded_layers.shape)
