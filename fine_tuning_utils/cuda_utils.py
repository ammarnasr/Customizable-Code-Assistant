import sys
import ctypes


# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def ConvertSMVer2ArchName(major, minor):
    # Returns the name of a CUDA architecture given a device capability
    # (major, minor) pair.
    # See _ConvertSMVer2ArchName in helper_cuda.h in NVIDIA's CUDA Samples.
    return {(1, 0): "Tesla",
            (1, 1): "Tesla",
            (1, 2): "Tesla",
            (1, 3): "Tesla",
            (2, 0): "Fermi",
            (2, 1): "Fermi",
            (3, 0): "Kepler",
            (3, 2): "Kepler",
            (3, 5): "Kepler",
            (3, 7): "Kepler",
            (5, 0): "Maxwell",
            (5, 2): "Maxwell",
            (5, 3): "Maxwell",
            (6, 0): "Pascal",
            (6, 1): "Pascal",
            (6, 2): "Pascal",
            (7, 0): "Volta",
            (7, 2): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 7): "Ampere",
            (8, 9): "Ampere",
            (9, 0): "Hopper",
            }.get((major, minor), "Unknown")


def ConvertSMVer2Cores(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {(1, 0): 8,    # Tesla
            (1, 1): 8,
            (1, 2): 8,
            (1, 3): 8,
            (2, 0): 32,   # Fermi
            (2, 1): 48,
            (3, 0): 192,  # Kepler
            (3, 2): 192,
            (3, 5): 192,
            (3, 7): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (5, 3): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (6, 2): 128,
            (7, 0): 64,   # Volta
            (7, 2): 64,
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 7): 128,
            (8, 9): 128,  # Ada
            (9, 0): 128,  # Hopper
            }.get((major, minor), 0)


def ConvertArchName2OpsPerSM(arch_name, precision):
    return {('Volta', 'FP64'): 2,
            ('Volta', 'FP32'): 64,
            ('Volta', 'FP16'): 128,
            ('Volta', 'INT8'): 256,
            ('Turing', 'FP64'): 2,
            ('Turing', 'FP32'): 64,
            ('Turing', 'FP16'): 128,
            ('Turing', 'INT8'): 256,
            ('Ampere', 'FP64'): 32,
            ('Ampere', 'FP32'): 128,
            ('Ampere', 'FP16'): 256,
            ('Ampere', 'INT8'): 512,
    }.get((arch_name, precision), 0)


def main():
    info_dict = {}
    libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + ' '.join(libnames))

    nGpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
        return 1
    print("Found %d device(s)." % nGpus.value)
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
            return 1
        print("Device: %d" % i)
        info_dict[f'device_{i}'] = {}
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            device_name = name.split(b'\0', 1)[0].decode()
            print("  Name: %s" % device_name)
            info_dict[f'device_{i}']['Name'] = device_name
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            compute_capability = f'{cc_major.value}.{cc_minor.value}'
            print("  Compute Capability: %s" % compute_capability)
            info_dict[f'device_{i}']['Compute Capability'] = compute_capability
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            multiprocessors = cores.value
            cuda_cores = cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value)
            cuda_arch = ConvertSMVer2ArchName(cc_major.value, cc_minor.value)
            print("  Multiprocessors: %d" % multiprocessors)
            print("  CUDA Cores: %s" % (cuda_cores or "unknown"))
            print("  CUDA Architecture: %s" % cuda_arch)
            info_dict[f'device_{i}']['Multiprocessors'] = multiprocessors
            info_dict[f'device_{i}']['CUDA Cores'] = cuda_cores
            info_dict[f'device_{i}']['CUDA Architecture'] = cuda_arch

            presicions = ['FP64', 'FP32', 'FP16', 'INT8']
            info_dict[f'device_{i}']['Ops per cycle'] = {}
            for precision in presicions:
                ops_per_cycle = ConvertArchName2OpsPerSM(cuda_arch, precision)
                print(f"  {precision} ops per cycle: {ops_per_cycle}")
                info_dict[f'device_{i}']['Ops per cycle'][precision] = ops_per_cycle

                

            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                concurrrent_threads = cores.value * threads_per_core.value
                print("  Concurrent threads: %d" % concurrrent_threads)
                info_dict[f'device_{i}']['Concurrent threads'] = concurrrent_threads
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            gpu_clockrate = clockrate.value / 1000.
            print("  GPU clock: %g MHz" % gpu_clockrate)
            info_dict[f'device_{i}']['GPU clock'] = gpu_clockrate
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
            memory_clockrate = clockrate.value / 1000.
            print("  Memory clock: %g MHz" % memory_clockrate)
            info_dict[f'device_{i}']['Memory clock'] = memory_clockrate
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                total_memory = totalMem.value / 1024**2
                free_memory = freeMem.value / 1024**2
                print("  Total Memory: %ld MiB" % total_memory)
                print("  Free Memory: %ld MiB" % free_memory)
                info_dict[f'device_{i}']['Total Memory (MiB)'] = total_memory
                info_dict[f'device_{i}']['Free Memory (MiB)'] = free_memory
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                print("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
            cuda.cuCtxDetach(context)
    return info_dict



def gpu_flops(gpu_info, precision=1):
    precision_map = {
        1: 'FP64',
        2: 'FP32',
        4: 'FP16',
    }
    clock_rate = gpu_info['GPU clock'] # in MHz
    clock_rate *= 1000**2 # in Hz
    cores = gpu_info['CUDA Cores']
    float_precision = precision_map[precision]
    flops_per_clock_cycle = gpu_info['Ops per cycle'][float_precision]
    flops = clock_rate * cores * flops_per_clock_cycle * precision
    tera_flops = flops/1000**4
    return tera_flops

if __name__=="__main__":
    sys.exit(main())