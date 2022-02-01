import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import numpy as np

from utils.SharedArray import SharedJaxParams

import multiprocessing as mp

import time

def test(array:SharedJaxParams):
    print(array.get())
    array.re_init()

    time.sleep(1)

    array.set(np.array([1]))

if __name__ == "__main__":
    mp.set_start_method("spawn")

    array = SharedJaxParams(np.array([2]))

    process = mp.Process(target=test, args=(array,))
    process.start()
    process.join()
    
    print(array.get())