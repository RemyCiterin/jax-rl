import faster_fifo 
import faster_fifo_reduction
from queue import Full, Empty
import queue

class Queue:
    def __init__(self, max_size=10, max_size_bytes=400 * 1000 * 1000, is_mp_queue=True):
        if is_mp_queue: self.queue = faster_fifo.Queue(max_size_bytes=max_size_bytes)
        else: self.queue = queue.Queue(max_size)
        self.is_mp_queue = is_mp_queue
        self.max_size = max_size

    def get(self, *args, **kargs):
        return self.queue.get(*args, **kargs)

    def put(self, obj, *args, **kargs):
        if self.is_mp_queue: 
            while self.queue.qsize() >= self.max_size: os.sched_yield()
        self.queue.put(obj, *args, **kargs)
