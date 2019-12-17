[WIP]

# Multilingual-Bert-as-a-service

1. Create a virtual python environment with virtualenv  
`virtualenv env`
2. Activate the virtual environment 
`source env/bin/activate`
3. Run `python mbert_server/server_controller.py` to set up the MBert server. This will set up each component of the server architecture in their correct order (ventilator, worker, sink). 
3. Run `python mbert_client/mbert_client.py` to activate the client that feeds sentences to the service. 
4. Wait for the sink to collect results! 

# About the service 

The service achitecture here follows the ventilator-worker-sink layout outlined in the ZeroMQ docs.  ![vent-worker](vent-worker.png)
