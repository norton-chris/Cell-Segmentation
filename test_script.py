import random
import ray

ray.init(num_cpus=4)


@ray.remote
def loop():
    for i in range(0, 10):
        print(i)


for i in range(0, 10):
    loop.remote()


