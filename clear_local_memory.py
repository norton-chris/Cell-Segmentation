from numba import cuda

device = cuda.get_current_device()
device.reset()
with cuda.gpus[0]:
    d_a = cuda.to_device(0)
    print(d_a)

print(cuda.list_devices())