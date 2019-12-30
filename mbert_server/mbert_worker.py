import sys
import time 
import zmq
from transformers import BertModel, BertTokenizer
from utils import send_array_and_str, recv_array_and_str, preprocess
import torch
from multiprocessing import Process


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

    def get_model(self):
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        model.eval()
        return model

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink_embed, sink_token, *receivers):
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink_embed.connect(self.sink_address)
        sink_token.connect(self.sink_address)
        for r in estimator.predict(self.input_fn_builder(receivers, tf, sink_token), yield_single_examples=False):
            send_ndarray(sink_embed, r['client_id'], r['encodes'], ServerCmd.data_embed)
            logger.info('job done\tsize: %s\tclient: %s' % (r['encodes'].shape, r['client_id']))

    def input_fn_builder(self, socks, tf, sink):
        from .bert.extract_features import convert_lst_to_features
        from .bert.tokenization import FullTokenizer

        def gen(self, socks, sink):
            # Windows does not support logger in MP environment, thus get a new logger
            # inside the process for better compatibility
            logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), self.verbose)
            tokenizer = FullTokenizer(vocab_file=os.path.join(self.model_dir, 'vocab.txt'), do_lower_case=self.do_lower_case)

            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)

            logger.info('ready and listening!')
            self.is_ready.set()

            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg), client_id))
                        # check if msg is a list of list, if yes consider the input is already tokenized
                        all(isinstance(el, list) for el in msg)
                        yield {
                            'client_id': client_id,
                            'input_ids': [f.input_ids for f in tmp_f],
                            'input_mask': [f.input_mask for f in tmp_f],
                            'input_type_ids': [f.input_type_ids for f in tmp_f]
                        }
        return input_fn


def run_worker():
    # set up zmq
    context = zmq.Context()
    # socket to receive work from ventilator
    work_receiver = context.socket(zmq.PULL)
    work_receiver.connect('tcp://localhost:5557')
    # socket to send results to sink
    results_sender = context.socket(zmq.PUSH)
    results_sender.connect('tcp://localhost:5558')
    print('bound sockets')
    # set up model 
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    print('loaded pre-trained model')
    model.eval()
    while True:
        sentence, tokens_tensor = recv_array_and_str(work_receiver)
        torch_tensor = torch.from_numpy(tokens_tensor)
        with torch.no_grad():
            outputs = model(torch_tensor)
            encoded_layers = outputs[0].numpy()
            send_array_and_str(results_sender, encoded_layers, sentence)
            print('sent response')

if __name__=='__main__':
    run_worker()
