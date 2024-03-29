#!/usr/bin/env python3

# A (probably not fully safe) cache for avoiding recomputation of
# dequantized tensors, when the threads are working on the same tensor.
#
# This only exists to avoid calling dequantize() a lot.
import threading
class ECache:
    def __init__(self):
        # cache is dictionary:
        # key ->   (value, refcount, computing)
        # <any> -> (<any>, int, bool)
        self.cache = {}
        self.lock = threading.Lock()
        self.cond = threading.Condition(lock=self.lock)  # used if a thread can't pick up work

    def with_values(self, keys, compute_func, action_func):
        if not keys:
            return action_func([])

        acquired = False
        while True:
            acquired = True
            self.lock.acquire()
            locked_keys = []
            vals = []
            waitables = []
            computables = []
            try:
                for key in keys:
                    if key in self.cache and self.cache[key][2] == False:
                        self.cache[key][1] += 1
                        locked_keys.append(key)
                        vals.append(self.cache[key][0])
                    elif key in self.cache:
                        self.cache[key][1] += 1
                        locked_keys.append(key)
                        waitables.append((len(vals), key))
                        vals.append(None)
                    else:
                        self.cache[key] = [None, 1, True]
                        locked_keys.append(key)
                        self.lock.release()
                        acquired = False

                        value = compute_func(key)
                        with self.lock:
                            self.cache[key][0] = value
                            self.cache[key][2] = False
                            # wake up threads that might be waiting for this
                            # key to be computed.
                            self.cond.notify_all()
                        vals.append(value)

                        acquired = True
                        self.lock.acquire()

                while True:
                    new_waitables = []
                    for idx, key in waitables:
                        if self.cache[key][2] == False:
                            vals[idx] = self.cache[key][0]
                            continue
                        new_waitables.append((idx, key))

                    waitables = new_waitables
                    if waitables:
                        self.cond.wait()
                    else:
                        break

                acquired = False
                self.lock.release()
                return action_func(vals)
            finally:
                if acquired:
                    self.lock.release()
                    acquired = False
                if locked_keys:
                    with self.lock:
                        for key in locked_keys:
                            self.cache[key][1] -= 1
                            if self.cache[key][1] == 0:
                                del self.cache[key]

    def with_value(self, key, compute_func, action_func):
        keys = [key]
        def action_func_wrapper(vals):
            return action_func(vals[0])

        return self.with_values(keys, compute_func, action_func_wrapper)

def unit_test():
    import concurrent.futures as cf
    import random
    import tqdm
    import time
    import sys

    ecache = ECache()

    n_keys = 1000
    keys = set(range(1000))

    def fake_worker():
        n_keys_to_use = random.randint(1, 50)
        work_takes_time = random.random() * 3.0

        work_keys = [random.randint(0, n_keys-1) for _ in range(n_keys_to_use)]

        def compute_common_thing(key):
            common_thing_compute_time = random.random() * 1.0
            time.sleep(common_thing_compute_time)
            return key + 12345

        def action(fake_computed_things):
            assert len(fake_computed_things) == n_keys_to_use
            for f in fake_computed_things:
                assert (f - 12345) in keys
            assert list(map(lambda x: x - 12345, fake_computed_things)) == work_keys
            time.sleep(work_takes_time)



        ecache.with_values(
            work_keys,
            compute_common_thing,
            action
        )

    n_tasks = 10000

    with cf.ThreadPoolExecutor(max_workers=200) as executor:
        futs = []
        for _ in range(n_tasks):
            futs.append(executor.submit(fake_worker))

        for fut in tqdm.tqdm(cf.as_completed(futs)):
            try:
                fut.result()
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.stderr.flush()
                raise

    assert ecache.cache == {}

if __name__ == '__main__':
    print('Running unit test')
    unit_test()
