import dataclasses
from typing import Dict, List

@dataclasses.dataclass
class GPUTensorCoreSpec:
    """Specifications of a GPU tensor core."""
    name: str
    clock_rate_ghz: float
    num_tensor_cores: int
    flops_per_cycle_fp32: int = 64
    flops_per_cycle_fp16: int = 128

@dataclasses.dataclass  
class GPUSpec:
    """Specifications of a GPU."""
    name: str
    architecture: str
    gpus: List[GPUTensorCoreSpec]

@dataclasses.dataclass
class HardwareSpec:
    """Specifications of available hardware."""
    gpus: List[GPUSpec]


def calculate_gpu_throughput(gpu: GPUTensorCoreSpec, 
                             precision: str = "fp32") -> float:
    """Calculate throughput of a GPU in TFLOPS."""
    flops_per_cycle = (gpu.flops_per_cycle_fp32 
                       if precision == "fp32" 
                       else gpu.flops_per_cycle_fp16)  
    throughput = (gpu.clock_rate_ghz * 1e9 * gpu.num_tensor_cores * 
                  flops_per_cycle)
    return throughput / 1e12


def calculate_aggregate_throughput(hardware: HardwareSpec,
                                   precision: str = "fp32") -> float:
    """Calculate aggregated throughput across GPUs in TFLOPS.""" 
    throughput = 0
    for gpu_spec in hardware.gpus:
        for gpu in gpu_spec.gpus:
            throughput += calculate_gpu_throughput(gpu, precision)
    return throughput


@dataclasses.dataclass   
class ExperimentSpec:
    """Specifications of a training experiment."""
    model_params: int  
    dataset: int
    hardware: HardwareSpec
    precision: str = "fp32"
    hours_trained: float = 1.0 # training hours
    

def calculate_compute_cost(spec: ExperimentSpec) -> float:
    """Calculate expected compute cost of an experiment."""
    
    throughput = calculate_aggregate_throughput(spec.hardware, 
                                                spec.precision)
                                                
    expected_compute = 6 * spec.model_params * spec.dataset  
    actual_compute = throughput * 1e12 * spec.hours_trained  
    
    return {
        "Required FLOPs": expected_compute / 1e12,
        "Actual FLOPs": actual_compute / 1e12,
    }

def main():
    # Define GPU specifications
    gpus = [
        GPUTensorCoreSpec(name="A100", clock_rate_ghz=1.41, 
                          num_tensor_cores=6912),
        GPUTensorCoreSpec(name="V100", clock_rate_ghz=1.53, 
                          num_tensor_cores=5120),
    ]
    gpu_specs = GPUSpec(name="NVIDIA", architecture="Ampere", gpus=gpus)
    
    # Define hardware specifications
    hardware = HardwareSpec(gpus=[gpu_specs])
    
    # Define experiment specifications
    spec = ExperimentSpec(model_params=1e9, dataset=1e12, 
                           hardware=hardware, precision="fp32", 
                           hours_trained=1.0)
    
    # Calculate compute cost
    cost = calculate_compute_cost(spec)
    print(f"Compute cost: {cost:.2f} TFLOPS")


















# import torch
# import pandas as pd
# import plotly.graph_objects as go
# import sys
# import ctypes

# # Some constants taken from cuda.h
# CUDA_SUCCESS = 0
# CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
# CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
# CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
# CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

# # GPU Database containing the information about the most common GPUs.
# GPU_DATABASE ={
#     "NVIDIA GeForce GTX 1650": {
#         "Name": "NVIDIA GeForce GTX 1650",
#         "Compute Capability": "7.5",
#         "Multiprocessors": 14,
#         "CUDA Cores": 896,
#         "CUDA Architecture": "Turing",
#         "Ops per cycle": {
#             "FP64": 2,
#             "FP32": 64,
#             "FP16": 128,
#             "INT8": 256
#         },
#         "Concurrent threads": 14336,
#         "GPU clock": 1515.0,
#         "Memory clock": 6001.0,
#         "Total Memory (MiB)": 4095.6875,
#     },
#     "NVIDIA T4":{
#         "Name": "NVIDIA T4",
#         "Compute Capability": "7.5",
#         "Multiprocessors": 40,
#         "CUDA Cores": 2560,
#         "CUDA Architecture": "Turing",
#         "Ops per cycle": {
#             "FP64": 2,
#             "FP32": 64,
#             "FP16": 128,
#             "INT8": 256
#         },
#         "Concurrent threads": 40960,
#         "GPU clock": 1590.0,
#         "Memory clock": 5001.0,
#         "Total Memory (MiB)": 15102.0625,
#     },
#     "NVIDIA Tesla P100-PCIE-16GB":{
#         "Name": "NVIDIA Tesla P100-PCIE-16GB",
#         "Compute Capability": "6.0",
#         "Multiprocessors": 56,
#         "CUDA Cores": 3584,
#         "CUDA Architecture": "Pascal",
#         "Ops per cycle": {
#             "FP64": 32,
#             "FP32": 128,
#             "FP16": 256,
#             "INT8": 512
#         },
#         "Concurrent threads": 100352,
#         "GPU clock": 1328.0,
#         "Memory clock": 715.0,
#         "Total Memory (MiB)": 16276.0,
#     },
#     "NVIDIA Tesla V100-SXM2-16GB":{
#         "Name": "NVIDIA Tesla V100-SXM2-16GB",
#         "Compute Capability": "7.0",
#         "Multiprocessors": 80,
#         "CUDA Cores": 5120,
#         "CUDA Architecture": "Volta",
#         "Ops per cycle": {
#             "FP64": 2,
#             "FP32": 64,
#             "FP16": 128,
#             "INT8": 256
#         },
#         "Concurrent threads": 409600,
#         "GPU clock": 1530.0,
#         "Memory clock": 877.0,
#         "Total Memory (MiB)": 16130.0,
#     },
#     "NVIDIA A100-SXM4-40GB":{
#         "Name": "NVIDIA A100-SXM4-40GB",
#         "Compute Capability": "8.0",
#         "Multiprocessors": 108,
#         "CUDA Cores": 6912,
#         "CUDA Architecture": "Ampere",
#         "Ops per cycle": {
#             "FP64": 32,
#             "FP32": 128,
#             "FP16": 256,
#             "INT8": 512
#         },
#         "Concurrent threads": 746496,
#         "GPU clock": 1410.0,
#         "Memory clock": 1556.0,
#         "Total Memory (MiB)": 40536.0,
#     },
# }

# def list_common_gpus():
#     return list(GPU_DATABASE.keys())

# def ConvertSMVer2ArchName(major, minor):
#     # Returns the name of a CUDA architecture given a device capability
#     # (major, minor) pair.
#     # See _ConvertSMVer2ArchName in helper_cuda.h in NVIDIA's CUDA Samples.
#     return {(1, 0): "Tesla",
#             (1, 1): "Tesla",
#             (1, 2): "Tesla",
#             (1, 3): "Tesla",
#             (2, 0): "Fermi",
#             (2, 1): "Fermi",
#             (3, 0): "Kepler",
#             (3, 2): "Kepler",
#             (3, 5): "Kepler",
#             (3, 7): "Kepler",
#             (5, 0): "Maxwell",
#             (5, 2): "Maxwell",
#             (5, 3): "Maxwell",
#             (6, 0): "Pascal",
#             (6, 1): "Pascal",
#             (6, 2): "Pascal",
#             (7, 0): "Volta",
#             (7, 2): "Volta",
#             (7, 5): "Turing",
#             (8, 0): "Ampere",
#             (8, 6): "Ampere",
#             (8, 7): "Ampere",
#             (8, 9): "Ampere",
#             (9, 0): "Hopper",
#             }.get((major, minor), "Unknown")

# def ConvertSMVer2Cores(major, minor):
#     # Returns the number of CUDA cores per multiprocessor for a given
#     # Compute Capability version. There is no way to retrieve that via
#     # the API, so it needs to be hard-coded.
#     # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
#     return {(1, 0): 8,    # Tesla
#             (1, 1): 8,
#             (1, 2): 8,
#             (1, 3): 8,
#             (2, 0): 32,   # Fermi
#             (2, 1): 48,
#             (3, 0): 192,  # Kepler
#             (3, 2): 192,
#             (3, 5): 192,
#             (3, 7): 192,
#             (5, 0): 128,  # Maxwell
#             (5, 2): 128,
#             (5, 3): 128,
#             (6, 0): 64,   # Pascal
#             (6, 1): 128,
#             (6, 2): 128,
#             (7, 0): 64,   # Volta
#             (7, 2): 64,
#             (7, 5): 64,   # Turing
#             (8, 0): 64,   # Ampere
#             (8, 6): 128,
#             (8, 7): 128,
#             (8, 9): 128,  # Ada
#             (9, 0): 128,  # Hopper
#             }.get((major, minor), 0)

# def ConvertArchName2OpsPerSM(arch_name, precision):
#     return {('Volta', 'FP64'): 2,
#             ('Volta', 'FP32'): 64,
#             ('Volta', 'FP16'): 128,
#             ('Volta', 'INT8'): 256,
#             ('Turing', 'FP64'): 2,
#             ('Turing', 'FP32'): 64,
#             ('Turing', 'FP16'): 128,
#             ('Turing', 'INT8'): 256,
#             ('Ampere', 'FP64'): 32,
#             ('Ampere', 'FP32'): 128,
#             ('Ampere', 'FP16'): 256,
#             ('Ampere', 'INT8'): 512,
#     }.get((arch_name, precision), 0)

# def cuda_main(verbose=True):
#     info_dict = {}
#     libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll', 'cuda.dll')
#     for libname in libnames:
#         try:
#             cuda = ctypes.CDLL(libname)
#         except OSError:
#             continue
#         else:
#             break
#     else:
#         raise OSError("could not load any of: " + ' '.join(libnames))
#     nGpus = ctypes.c_int()
#     name = b' ' * 100
#     cc_major = ctypes.c_int()
#     cc_minor = ctypes.c_int()
#     cores = ctypes.c_int()
#     threads_per_core = ctypes.c_int()
#     clockrate = ctypes.c_int()
#     freeMem = ctypes.c_size_t()
#     totalMem = ctypes.c_size_t()

#     result = ctypes.c_int()
#     device = ctypes.c_int()
#     context = ctypes.c_void_p()
#     error_str = ctypes.c_char_p()

#     result = cuda.cuInit(0)
#     if result != CUDA_SUCCESS:
#         cuda.cuGetErrorString(result, ctypes.byref(error_str))
#         print("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
#         return 1
#     result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
#     if result != CUDA_SUCCESS:
#         cuda.cuGetErrorString(result, ctypes.byref(error_str))
#         print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
#         return 1
#     if verbose:
#         print("Found %d device(s)." % nGpus.value)
#     for i in range(nGpus.value):
#         result = cuda.cuDeviceGet(ctypes.byref(device), i)
#         if result != CUDA_SUCCESS:
#             cuda.cuGetErrorString(result, ctypes.byref(error_str))
#             print("cuDeviceGet failed with error code %d: %s" % (result, error_str.value.decode()))
#             return 1
#         if verbose:
#             print("Device: %d" % i)
#         info_dict[f'device_{i}'] = {}
#         if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
#             device_name = name.split(b'\0', 1)[0].decode()
#             if verbose:
#                 print("  Name: %s" % device_name)
#             info_dict[f'device_{i}']['Name'] = device_name
#         if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
#             compute_capability = f'{cc_major.value}.{cc_minor.value}'
#             if verbose:
#                 print("  Compute Capability: %s" % compute_capability)
#             info_dict[f'device_{i}']['Compute Capability'] = compute_capability
#         if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
#             multiprocessors = cores.value
#             cuda_cores = cores.value * ConvertSMVer2Cores(cc_major.value, cc_minor.value)
#             cuda_arch = ConvertSMVer2ArchName(cc_major.value, cc_minor.value)
#             if verbose:
#                 print("  Multiprocessors: %d" % multiprocessors)
#                 print("  CUDA Cores: %s" % (cuda_cores or "unknown"))
#                 print("  CUDA Architecture: %s" % cuda_arch)
#             info_dict[f'device_{i}']['Multiprocessors'] = multiprocessors
#             info_dict[f'device_{i}']['CUDA Cores'] = cuda_cores
#             info_dict[f'device_{i}']['CUDA Architecture'] = cuda_arch

#             presicions = ['FP64', 'FP32', 'FP16', 'INT8']
#             info_dict[f'device_{i}']['Ops per cycle'] = {}
#             for precision in presicions:
#                 ops_per_cycle = ConvertArchName2OpsPerSM(cuda_arch, precision)
#                 if verbose:
#                     print(f"  {precision} ops per cycle: {ops_per_cycle}")
#                 info_dict[f'device_{i}']['Ops per cycle'][precision] = ops_per_cycle

                

#             if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
#                 concurrrent_threads = cores.value * threads_per_core.value
#                 if verbose:
#                     print("  Concurrent threads: %d" % concurrrent_threads)
#                 info_dict[f'device_{i}']['Concurrent threads'] = concurrrent_threads
#         if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
#             gpu_clockrate = clockrate.value / 1000.
#             if verbose:
#                 print("  GPU clock: %g MHz" % gpu_clockrate)
#             info_dict[f'device_{i}']['GPU clock'] = gpu_clockrate
#         if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
#             memory_clockrate = clockrate.value / 1000.
#             if verbose:
#                 print("  Memory clock: %g MHz" % memory_clockrate)
#             info_dict[f'device_{i}']['Memory clock'] = memory_clockrate
#         try:
#             result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
#         except AttributeError:
#             result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
#         if result != CUDA_SUCCESS:
#             cuda.cuGetErrorString(result, ctypes.byref(error_str))
#             print("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
#         else:
#             try:
#                 result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
#             except AttributeError:
#                 result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
#             if result == CUDA_SUCCESS:
#                 total_memory = totalMem.value / 1024**2
#                 free_memory = freeMem.value / 1024**2
#                 if verbose:
#                     print("  Total Memory: %ld MiB" % total_memory)
#                     print("  Free Memory: %ld MiB" % free_memory)
#                 info_dict[f'device_{i}']['Total Memory (MiB)'] = total_memory
#                 # info_dict[f'device_{i}']['Free Memory (MiB)'] = free_memory
#             else:
#                 cuda.cuGetErrorString(result, ctypes.byref(error_str))
#                 print("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
#             cuda.cuCtxDetach(context)
#     return info_dict


# def gpu_flops(gpu_info, precision=1):
#     precision_map = {
#         1: 'FP64',
#         2: 'FP32',
#         4: 'FP16',
#     }
#     clock_rate = gpu_info['GPU clock'] # in MHz
#     clock_rate *= 1000**2 # in Hz
#     cores = gpu_info['CUDA Cores']
#     float_precision = precision_map[precision]
#     flops_per_clock_cycle = gpu_info['Ops per cycle'][float_precision]
#     flops = clock_rate * cores * flops_per_clock_cycle * precision
#     tera_flops = flops/1000**4
#     return tera_flops

# def custom_gpu_info(gpu_name):
#     if gpu_name in GPU_DATABASE:
#         return GPU_DATABASE[gpu_name]
#     else:
#         print(f'GPU {gpu_name} not found in database')
#         return None
    
# def print_large_number(number, bytebased=False):
#     '''
#     Utility function to return a string representation of a large number
#     It uses scales (Trillion, Billion, Million, Thousand)
#     If bytebased is True, it uses scales (Terabyte, Gigabyte, Megabyte, Kilobyte)
#     '''
#     if bytebased:
#         scales = ['TB', 'GB', 'MB', 'KB']
#     else:
#         scales = ['T', 'B', 'M', 'K']
#     scale = 0
#     while number >= 1000:
#         number /= 1000
#         scale += 1
#         if scale == 3:
#             break
#     return f'{number:.2f}{scales[scale]}'


# def local_gpu_info():
#     if torch.cuda.is_available():
#         local_cuda_info = cuda_main(verbose=False)
#         num_gpus = len(local_cuda_info)
#         print(f'GPU is available! You have {num_gpus} GPUs Locally.')
#     else:
#         print('No GPU available Locally!')
#         local_cuda_info = None
#         num_gpus = 0
#     return local_cuda_info, num_gpus




# def number_of_tokens_estimator(dataset_size):
#     tokens_per_gb = 90 * 10**6
#     if 'TB' in dataset_size:
#         modifier = 10**12
#         val = float(dataset_size.replace('TB', ''))
#     elif 'GB' in dataset_size:
#         modifier = 10**9
#         val = float(dataset_size.replace('GB', ''))
#     elif 'MB' in dataset_size:
#         modifier = 10**6
#         val = float(dataset_size.replace('MB', ''))
#     elif 'KB' in dataset_size:
#         modifier = 10**3
#         val = float(dataset_size.replace('KB', ''))
#     else:
#         modifier = 1
#         val = float(dataset_size)
#     return int(val * modifier * tokens_per_gb / 10**9)
    




# def Calculate_throughput(gpu_info, float_precision='FP32'):
#     precision_map = {
#         'FP64': 1,
#         'FP32': 2,
#         'FP16': 4,
#     }
#     clock_rate = gpu_info['GPU clock'] # in MHz
#     clock_rate *= 1000**2 # in Hz
#     cores = gpu_info['CUDA Cores']
#     precision = precision_map[float_precision]
#     flops_per_clock_cycle = gpu_info['Ops per cycle'][float_precision]
#     flops = clock_rate * cores * flops_per_clock_cycle * precision
#     tera_flops = flops/1000**4
#     return tera_flops


# def calculate_tao_for_gpu_precision(devices_info, precision='FP32'):
#     tao = 0
#     for device_info in devices_info:
#         tao += Calculate_throughput(device_info, precision=precision)
#     return tao



# def calculate_dataset_size(k=0):
#     num_tokens = number_of_tokens_estimator(dataset_size)
#     return num_tokens


# def calculate_training_time(k=0):
#     #Let the user enter the training time manually
#     training_time = st.number_input(
#         'Enter the training time in hours:',
#         value=1,
#         key=k,
#         step=10,
#     )
#     k += 1
#     #Display the training time
#     st.markdown(f"##### Training Time = :violet[{training_time} hours]")

#     #Add training time to the session state
#     st.session_state['training_time'] = training_time

#     return training_time*3600



# def main(num_parameters, training_time, dataset_size, gpu_info=None, local_cuda=False, precision='FP32', scaling_factor='Model Size'):
#     P = num_parameters
#     T = training_time
#     D = number_of_tokens_estimator(dataset_size)
#     if local_cuda:
#         local_cuda_info, num_gpus = local_gpu_info()
#         if num_gpus > 0:
#             tao = calculate_tao_for_gpu_precision(local_cuda_info, precision=precision)
#         else:
#             print('No GPU available Locally, try to use gpu_info parameter to pass the GPU info')
#             tao = 0
#     else:
#         tao = calculate_tao_for_gpu_precision(gpu_info, precision=precision)


#     if D >= 20*P:
#         print(f" The dataset size is sufficient for the model parameters (D >= 20P).")
#     else:
#         print(f" The dataset size is not sufficient for the model parameters (D < 20P).")

#     print("Now we can calculate the cost using the scaling law equation:")
#     print("C = tao*T = 6*P*D")
#     print(f"tao is the Aggregated Throughput = {tao} TFLOPS")
#     print(f"T is the Training Time = {T} Seconds")
#     print(f"P is the Number of Parameters = {P} Parameters")
#     print(f"D is the Number of Tokens = {D} Tokens")
#     print(f"6 is the Scaling Law Constant")
#     lhs_cost = (tao*(1000**4))*T
#     rhs_cost = 6*P*D
#     print(f"The FLOPs based on the Hardware and Training Time = {lhs_cost} TFLOPS")
#     print(f"The FLOPs based on the Model and Dataset Size = {rhs_cost} TFLOPS")

#     if lhs_cost > rhs_cost:
#         overtrain = lhs_cost - rhs_cost
#         optimal_hours = (rhs_cost/(tao*1000**4))/3600
#         optimal_throughput = (rhs_cost/T)/10**12
#         print(f"Overtraining = {overtrain} TFLOPS")
#         print(f"You can reduce either the training time or Hardware throughput to reduce the monetary cost.")
#         print(f"Optimal Throughput = {optimal_throughput} TFLOPS")
#         print(f"Optimal Training Time = {optimal_hours} hour(s)")
#     else:
#         undertrain = rhs_cost - lhs_cost
#         optimal_hours = (rhs_cost/(tao*1000**4))/3600
#         print(f"Undertraining = {undertrain} TFLOPS")
#         print(f"To train the model with the parameters, dataset size, and hardware you have selected, you need to train for at least {optimal_hours} hour(s)")
        


#     print(f"---")
#     print("<h1 style='text-align: center; color: Tomato;'>Compute Optimal LLMs Fine Tuning</h1>", unsafe_allow_html=True)
#     print("**We use the following multiplicative joint scaling law for LLM finetuning:**")
#     print("L(X, D) = A \left( \frac{1}{X^\alpha} \right) \left( \frac{1}{D_f^\beta} \right) + E")
#     print('''
#                 **Where:**

#                     - L(X,D) represents the loss function, which quantifies the difference between the predicted output and the desired output.
                
#                     - X and D are vectors of scaling factors and the fine-tuning data size, respectively. The scaling factors represent the magnitudes of the resources used during training, such as the amount of compute or the number of training examples.
                
#                     - A, E, α, and β are the parameters of the model. A represents the base loss, E represents the constant error, α and β represent the exponents of the scaling factors.
                
#                 ''')
    
