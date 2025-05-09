# DataEnrichment_DeepSort_SydaLab__Testbed

The python 3 should be installed in both edge and far-edge devices

    Run client.py on edge device with -> python client.py

    Run server_deepsort.py on far-edge device with -> python server_deepsort.py

    Note: The Host IP should be changed in both source code

# use GPU
conda install numba 
conda install cudatoolkit


# check if it is available
import GPUtil
GPUtil.getAvailable()
or
import torch
torch.cuda.is_available()
torch.cuda.get_device_properties(0)

