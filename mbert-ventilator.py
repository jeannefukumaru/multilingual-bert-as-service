import zmq 
import random 
import time 

context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.bind("tcp://*:5557")

sink = context.socket(zmq.PUSH)
sink.connect("tcp://localhost:5558")

print("Press Enter when the workers are ready: ")
_ = input()
print("sending tasks to workers...")

sink.send(b'0')

random.seed()

total_msec = 0
for task_nbr in range(100):
    workload = random.randint(1, 100)
    total_msec += workload
    
    sender.send_string(u'%i' % workload)
    
print("Total expected cost: %s msec" % total_msec)

time.sleep(1)
