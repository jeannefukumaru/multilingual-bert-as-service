import zmq 
import random 
import time 
import numpy as np 
import time 
import torch 
from utils import send_array_and_str, recv_array_and_str, preprocess, auto_bind, set_logger
from argparse import ArgumentParser 
import zmq.decorators as zmqd 
from zmq.utils import jsonapi 
import threading
import argparse 

class MbertVentilator(threading.Thread): 
    def __init__(self, args):
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.num_worker = args.num_worker
        self.num_concurrent_socket = max(8, args.num_worker * 2)
        self.port = args.port
        self.processes = []
        self.logger = set_logger(colored('VENTILATOR', 'magenta'), args.verbose)

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUSH)
    def run(self, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always the highest priority 
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets' % len(addr_backend_list))

        # start the sink process 
        self.logger.info('start the sink')
        proc_sink = BertSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start worker process
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = BertWorker(addr_backend_list, addr_sink, device_id)
            self.processes.append(process)
            process.start()

        rand_backend_socket = None 

        for p in self.processes:
            p.is_ready().wait()

        self.is_ready.set()
        self.logger.info('all set, ready to serve request!')

        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
                assert req_id.isdigit()
                assert msg_len.isdigit()
            except (ValueError, AssertionError):
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' %len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) in enumerate(request), exc_info=True))
            else:
                self.logger.info('new encode request\treq id: %d\tsize: %d\tclient: %s' % (int(req_id), int(msg_len), client))
                # register new job at sink 
                sink.send_multipart([client, msg_len, req_id])

                # renew the backend socket to prevent large job queueing up
                # [0] is reserved for high priority job
                # last used backennd shouldn't be selected either as it may be queued up already
                rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                # push a new job, note super large job will be pushed to one socket only,
                # leaving other sockets free
                job_id = client + b'#' + req_id
                if int(msg_len) > self.max_batch_size:
                    seqs = jsonapi.loads(msg)
                    job_gen = ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in
                                range(0, int(msg_len), self.max_batch_size))
                    for partial_job_id, job in job_gen:
                        push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                else:
                    push_new_job(job_id, msg, int(msg_len))

        for p in self.processes:
            p.close()
        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker),
                                                maxMemory=0.9, maxLoad=0.9)
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map

    # # zmq setup 
    # context = zmq.Context()
    # # bind ventilator to client to accept sentence strings 
    # client = context.socket(zmq.PULL)
    # client.bind('tcp://127.0.0.1:5555')
    # print('client bound')
    # # bind ventilator to worker to allocate jobs
    # sender = context.socket(zmq.PUSH)
    # sender.bind('tcp://127.0.0.1:5557')
    # print('sender bound')
    # # bind ventilator to sink
    # sink = context.socket(zmq.PUSH).bind('tcp://127.0.0.1:5558')
    # print('sink bound')
    # # give everything a second to spin up and bind
    # time.sleep(5)
    # print('ventilator is starting')
    # print('waiting for incoming messages...')
    # while True:
    #     sentence, tokens_tensor = recv_array_and_str(client)
    #     print("received request")
    #     print(sentence, tokens_tensor)
    #     send_array_and_str(sender, tokens_tensor, sentence)
    #     print("sent request")
    #     time.sleep(1)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument()
    MbertVentilator().run()
