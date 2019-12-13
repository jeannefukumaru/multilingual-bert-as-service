
from mbert_worker import run_worker
from mbert_sink import run_sink
from mbert_ventilator import run_ventilator
from mbert_client import client
from multiprocessing import Process

if __name__=='__main__':

    # create pool of workers to distribute work to 
    # worker_pool = range(3)
    # for w in worker_pool:
    #     Process(target=run_worker).start()
    #     print('started worker')
    
    client = Process(target=client)
    print('started client')
    sink = Process(target=run_sink)
    print('started sink')
    ventilator = Process(target=run_ventilator)
    print('started ventilator')



