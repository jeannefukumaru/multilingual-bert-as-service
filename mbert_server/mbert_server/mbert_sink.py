import sys
import time 
import zmq
import zmq.decorators as zmqd
from utils import *
from multiprocessing import Process
import multiprocessing
import numpy as np 
from zmq.utils import jsonapi

class ServerCmd:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    show_status = b'SHOW_STATUS'
    new_job = b'REGISTER'
    data_token = b'TOKENS'
    data_embed = b'EMBEDDINGS'
    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCmd).items())

class MbertSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out 
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), args.verbose)
        self.front_sink_addr = front_sink_addr
        self.max_seq_len = args.max_seq_len
        self.is_read = multiprocessing.Event()

    def close(self):
        self.logger.info('shutting down...')
        self.is_ready.clear()
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_jobs = defaultdict(lambda: SinkJob(self.show_tokens_to_client))

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend 
        frontend.send(receiver_addr.encode('ascii'))

        logger = set_logger(colored('SINK','green'), self.verbose)
        logger.info('ready')
        self.is_ready.set()

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                # parsing job_id and partial_id 
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = int(job_info[1] if len(job_info) == 2 else 0)
                server_cmd = ServerCmd()
                if msg[3] == ServerCmd.data_embed:
                    # parsing the ndarray
                    arr_info, arr_val = jsonapi.loads(msg[1]), msg[2]
                    x = np.frombuffer(memoryview(arr_val), dtype=arr_info['dtype']).reshape(arr_info['shape'])
                    pending_jobs[job_id].add_embed(x, partial_id)
                elif msg[3] == ServerCmd.data_token:
                    x = jsonapi.loads(msg[1])
                    pending_jobs[job_id].add_token(x, partial_id)
                else:
                    logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(msg))
                    logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(msg)), exc_info=True)

                logger.info('collect %s %s (E:%d/T:%d/A:%d)' % (msg[3], job_id,
                                                                pending_jobs[job_id].progress_embeds,
                                                                pending_jobs[job_id].progress_tokens,
                                                                pending_jobs[job_id].checksum))

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCmd.new_job:
                    job_info = client_addr + b'#' + req_id
                    # register a new job
                    pending_jobs[job_info].checksum = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                    if len(pending_jobs[job_info]._pending_embeds)>0 \
                            and pending_jobs[job_info].final_ndarray is None:
                        pending_jobs[job_info].add_embed(None, 0)
                elif msg_type == ServerCmd.show_config or msg_type == ServerCmd.show_status:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])

            # check if there are finished jobs, then send it back to workers
            finished = [(k, v) for k, v in pending_jobs.items() if v.is_done]
            for job_info, tmp in finished:
                client_addr, req_id = job_info.split(b'#')
                x, x_info = tmp.result
                sender.send_multipart([client_addr, x_info, x, req_id])
                logger.info('send back\tsize: %d\tjob id: %s' % (tmp.checksum, job_info))
                # release the job
                tmp.clear()
                pending_jobs.pop(job_info)


class SinkJob():
    def __init__(self, with_tokens=True):
        self._pending_embeds = []
        self.tokens = []
        self.tokens_ids =[]
        self.checksum = 0
        self.final_ndarray = None
        self.progress_tokens = 0
        self.progress_embeds = 0
        self.with_tokens = with_tokens 

    def clear(self):
        self._pending_embeds.clear()
        self.tokens_ids.clear()
        self.tokens.clear()
        del self.final_ndarray

    def _insert(self, data, pid, data_lst, idx_lst):
        lo = 0
        hi = len(idx_lst)
        while lo < hi:
            mid = (lo + hi) //2
            if pid < idx_lst[mid]:
                hi = mid 
            else:
                lo = mid + 1
        idx_lst.insert(lo, pid)
        data_lst.insert(lo, data)

    def add_embed(self, data, pid):
        '''monitor embeddings coming in, gather full or intermediate results into final array'''
        def fill_data():
            # self.final_ndarray = data
            self.final_ndarray[pid:(pid + data.shape[0]), 0:data.shape[1:]] = data 
            self.progress_embeds += progress
            if data is not None: # data has been received 
                progress = data.shape[0] # register progress 
            else: 
                progress = 0
            if not self.checksum: # if job hasn't finished
                self._pending_embeds.append((data, pid, progress)) # append data to an intermediate list 
            else:
                if self.final_ndarray is None:
                    if data is not None: # when job finish msg come to SINK earlier than job register
                        d_shape = list(data.shape[1:])
                    else:
                        d_shape = list(self._pending_embeds[0][0].shape[1:])
                    if data is not None:
                        dtype = data.dtype
                    else:
                        dtype = self._pending_embeds[0][0].dtype   
                self.final_ndarray = np.zeros([self.checksum] + d_shape, dtype=dtype)
            if data is not None:
                fill_data()
            while self._pending_embeds:
                data, pid, progress = self._pending_embeds.pop()
                fill_data()

        def add_token(self, data, pid):
            progress = len(data)
            self._insert(data, pid, self.tokens, self.tokens_ids)
            self.progress_tokens += progress 

        @property
        def is_done(self):
            if self.with_tokens:
                return self.checksum > 0 and self.checksum == self.progress_tokens and self.checksum == self.progress_embeds
            return self.checksum > 0 and self.checksum == self.progress_embeds 

        @property
        def result(self):
            x = self.final_ndarray
            x_info = {'dtype': str(x.dtype),
                    'shape': x.shape,
                    'tokens':list(chain.from_iterable(self.rokens)) if self.with_tokens else ''}
            x_info = jsonapi.dumps(x_info)
            return x, x_info


