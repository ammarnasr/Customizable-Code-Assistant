import streamlit as st
import cuda_utils
import torch
import pandas as pd
import plotly.graph_objects as go






def print_large_number(number, bytebased=False):
    '''
    Utility function to return a string representation of a large number
    It uses scales (Trillion, Billion, Million, Thousand)
    If bytebased is True, it uses scales (Terabyte, Gigabyte, Megabyte, Kilobyte)
    '''
    if bytebased:
        scales = ['TB', 'GB', 'MB', 'KB']
    else:
        scales = ['T', 'B', 'M', 'K']
    scale = 0
    while number >= 1000:
        number /= 1000
        scale += 1
        if scale == 3:
            break

    return f'{number:.2f}{scales[scale]}'


def add_local_gpu_info():
    #Check if GPU is available and add its info to the session state
    if torch.cuda.is_available():
        local_cuda_info = cuda_utils.main(verbose=False)
        num_gpus = len(local_cuda_info)
        st.success(f'GPU is available! You have {num_gpus} GPUs.')

        st.session_state['local_cuda_info'] = local_cuda_info
        st.session_state['num_local_gpus'] = num_gpus
    else:
        st.warning('No GPU available Localy!')


def compare_common_gpu_vs_local_gpu(k=0):
    # Get common gpus names for which we have info
    common_gpus = cuda_utils.list_common_gpus()
    #Display common gpus info and local gpus info side by side
    local_gpu_col, common_gpu_col = st.columns([1,1])
    with local_gpu_col:
        local_cuda_info = st.session_state['local_cuda_info']
        local_devices = local_cuda_info.keys()
        #Let the user select a local gpu device
        selected_local_device = st.selectbox(
            'Select a local GPU device:',
            list(local_devices),
            index=0,
            key=k
        )
        k += 1
        #Display the selected local gpu info
        st.write('Selected local GPU info:')
        st.write(local_cuda_info[selected_local_device])
    with common_gpu_col:
        #Let the user select a common gpu device
        selected_common_device = st.selectbox(
            'Select a common GPU device:',
            common_gpus,
            index=1,
            key=k
        )
        k += 1
        #Display the selected common gpu info
        st.write('Selected common GPU info:')
        st.write(cuda_utils.custome_gpu_info(selected_common_device))

def compare_two_common_gpus(k=0):
    # Get common gpus names for which we have info
    common_gpus = cuda_utils.list_common_gpus()
    
    col1, col2 = st.columns([1,1])

    with col1:
        first_common_device = st.selectbox(
            'Select the first common GPU device:',
            common_gpus,
            index=0,
            key=k
        )
        k += 1
        st.write('First common GPU info:')
        st.write(cuda_utils.custome_gpu_info(first_common_device))
    with col2:
        second_common_device = st.selectbox(
            'Select the second common GPU device:',
            common_gpus,
            index=1,
            key=k
        )
        k += 1
        st.write('Second common GPU info:')
        st.write(cuda_utils.custome_gpu_info(second_common_device))

def code_llms_info(k=0):
    code_llms_df = pd.read_csv('llms_info.csv')
    models = code_llms_df['Model'].unique()
    #Let the user select a model
    selected_model = st.selectbox(
        'Select a model:',
        models,
        index=0,
        key=k
    )
    k += 1
    #Get the model's info
    model_info = code_llms_df[code_llms_df['Model'] == selected_model]
    # Date,Model,Arch.,Size,Vocab,Context,Init. from,Dataset,Training,PL
    model_info = {
        'Date of publication': model_info['Date'].values[0],
        'Model': model_info['Model'].values[0],
        'Architecture': model_info['Arch.'].values[0],
        'Size': model_info['Size'].values[0],
        'Vocabulary': model_info['Vocab'].values[0],
        'Context': model_info['Context'].values[0],
        'Initilized from': model_info['Init. from'].values[0],
        'Dataset Size': model_info['Dataset'].values[0],
        'Training Size': model_info['Training'].values[0],
        'Number of Programming Languages': model_info['PL'].values[0],
    }
    #Display the model's info
    st.write('Model info:')
    st.write(model_info)

def compare_llms_models(k=0):
    code_llms_df = pd.read_csv('llms_info.csv')
    models = code_llms_df['Model'].unique()
    col1, col2 = st.columns([1,1])
    with col1:
        first_model = st.selectbox(
            'Select the first model:',
            models,
            index=0,
            key=k
        )
        k += 1
    
    with col2:
        second_model = st.selectbox(
            'Select the second model:',
            models,
            index=1,
            key=k
        )
        k += 1

    models_selected = [first_model, second_model]
    selected_code_llms_df = code_llms_df[code_llms_df['Model'].isin(models_selected)]
    selected_code_llms_df = selected_code_llms_df.reset_index(drop=True)
    st.write('Selected models info:')
    st.write(selected_code_llms_df)

def number_of_tokens_estimator(dataset_size):
    tokens_per_gb = 90 * 10**6
    if 'TB' in dataset_size:
        modifier = 10**12
        val = float(dataset_size.replace('TB', ''))
    elif 'GB' in dataset_size:
        modifier = 10**9
        val = float(dataset_size.replace('GB', ''))
    elif 'MB' in dataset_size:
        modifier = 10**6
        val = float(dataset_size.replace('MB', ''))
    elif 'KB' in dataset_size:
        modifier = 10**3
        val = float(dataset_size.replace('KB', ''))
    else:
        modifier = 1
        val = float(dataset_size)
    return int(val * modifier * tokens_per_gb / 10**9)
    


def code_datasets_info(k=0):
    code_datasets_df = pd.read_csv('code_datasets.csv')
    datasets = code_datasets_df['Dataset'].unique()
    #Let the user select a dataset
    selected_dataset = st.selectbox(
        'Select a dataset:',
        datasets,
        index=0,
        key=k
    )
    k += 1
    #Get the dataset's info
    dataset_info = code_datasets_df[code_datasets_df['Dataset'] == selected_dataset]
    # Dataset,PL,Size,Context,Source,URL
    dataset_info = {
        'Dataset': dataset_info['Dataset'].values[0],
        'Number of Programming Languages': dataset_info['PL'].values[0],
        'Size': dataset_info['Size'].values[0],
        'Description': dataset_info['Description'].values[0],
    }
    num_tokens = number_of_tokens_estimator(dataset_info['Size'])
    dataset_info['Estimated Number of Tokens'] = f'{num_tokens/10**9}B'
    #Display the dataset's info
    st.write('Dataset info:')
    st.write(dataset_info)

def compare_code_datasets(k=0):
    code_datasets_df = pd.read_csv('code_datasets.csv')
    datasets = code_datasets_df['Dataset'].unique()
    col1, col2 = st.columns([1,1])
    with col1:
        first_dataset = st.selectbox(
            'Select the first dataset:',
            datasets,
            index=0,
            key=k
        )
        k += 1
    
    with col2:
        second_dataset = st.selectbox(
            'Select the second dataset:',
            datasets,
            index=1,
            key=k
        )
        k += 1

    datasets_selected = [first_dataset, second_dataset]
    selected_code_datasets_df = code_datasets_df[code_datasets_df['Dataset'].isin(datasets_selected)]
    selected_code_datasets_df = selected_code_datasets_df.reset_index(drop=True)
    st.write('Selected datasets info:')
    st.write(selected_code_datasets_df)

def Calculate_throughput(gpu_info, precision=1):
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





def calculate_tao_for_gpu_precision(k=0):
    local_cuda_info = st.session_state['local_cuda_info']
    local_devices = local_cuda_info.keys()
    common_devices = cuda_utils.list_common_gpus()
    all_devices = list(local_devices) + common_devices
    #Let the user select gpu devices setup
    selected_devices = st.multiselect(
        'Select GPU devices setup:',
        all_devices,
        default=[all_devices[0], all_devices[1]],
        key=k
    )
    k += 1
    #Let the user select a precision
    precisions = ['1-FP64', '2-FP32', '4-FP16']
    selected_precision = st.selectbox(
        'Select a precision:',
        precisions,
        index=1,
        key=k
    )
    k += 1
    #Get the selected devices info
    selected_devices_info = []
    for device in selected_devices:
        if device in local_devices:
            selected_devices_info.append(local_cuda_info[device])
        else:
            selected_devices_info.append(cuda_utils.custome_gpu_info(device))
    #Calculate the aggregated throughput
    tao = 0
    for device_info in selected_devices_info:
        tao += Calculate_throughput(device_info, precision=int(selected_precision[0]))
    #Display the aggregated throughput
    st.markdown(f"##### Aggregated Throughput = :violet[{tao :.2f} TFLOPS]")

    #Add Selected devices info to the session state
    st.session_state['selected_devices_info'] = selected_devices_info
    st.session_state['selected_precision'] = selected_precision

    return tao

def calcualte_model_parameters(k=0):
    #Let the user select a model or enter the number of parameters manually
    code_llms_df = pd.read_csv('llms_info.csv')
    models = code_llms_df['Model'].unique()

    st.info(f" You can select a model or enter the number of parameters manually.")
    
    add_paramters_manually = st.checkbox(
        'Add parameters manually',
        value=False,
        key=k
    )
    k += 1
    if add_paramters_manually:
        num_parameters = st.number_input(
            'Enter the number of parameters:',
            value=10**9,
            key=k,
            step=10**6
        )
        k += 1
    else:
        selected_model = st.selectbox(
            'Select a model:',
            models,
            index=0,
            key=k
        )
        k += 1
        #Get the model's info
        model_info = code_llms_df[code_llms_df['Model'] == selected_model]
        num_parameters = model_info['Size'].values[0]
        if 'M' in num_parameters:
            num_parameters = float(num_parameters.replace('M', ''))
            num_parameters *= 10**6
        elif 'B' in num_parameters:
            num_parameters = float(num_parameters.replace('B', ''))
            num_parameters *= 10**9
        else:
            num_parameters = float(num_parameters)
    #Display the number of parameters
    if num_parameters > 10**9:
        num_parameters_str = f'{num_parameters/10**9}B'
    elif num_parameters > 10**6:
        num_parameters_str = f'{num_parameters/10**6}M'
    else:
        num_parameters_str = f'{num_parameters}'
    st.markdown(f"##### Number of Parameters = :violet[{num_parameters_str}]")

    #Add Model info to the session state
    if add_paramters_manually:
        model_info = [{'Model':'Custom Model', 'Size':num_parameters_str}]
        st.session_state['selected_model_info'] = pd.DataFrame(model_info)
    else:
        st.session_state['selected_model_info'] = model_info

    return num_parameters

def calculate_dataset_size(k=0):
    #Let the user select a dataset or enter the dataset size manually
    code_datasets_df = pd.read_csv('code_datasets.csv')
    datasets = code_datasets_df['Dataset'].unique()

    add_dataset_size_manually = st.checkbox(
        'Add dataset size manually',
        value=False,
        key=k
    )
    k += 1
    if add_dataset_size_manually:
        dataset_size = st.text_input(
            'Enter the dataset size:',
            value='100GB',
            key=k
        )
        k += 1
    else:
        selected_dataset = st.selectbox(
            'Select a dataset:',
            datasets,
            index=0,
            key=k
        )
        k += 1
        #Get the dataset's info
        dataset_info = code_datasets_df[code_datasets_df['Dataset'] == selected_dataset]
        dataset_size = dataset_info['Size'].values[0]
    #Display the dataset size and the number of tokens
    num_tokens = number_of_tokens_estimator(dataset_size)
    if num_tokens > 10**9:
        num_tokens_str = f'{num_tokens/10**9}B'
    elif num_tokens > 10**6:
        num_tokens_str = f'{num_tokens/10**6}M'
    else:
        num_tokens_str = f'{num_tokens}'
    st.markdown(f"##### Dataset Size = :violet[{dataset_size}] (Estimated Number of Tokens = :violet[{num_tokens_str}])")

    #Add Dataset info to the session state
    if add_dataset_size_manually:
        dataset_info = [{'Dataset':'Custom Dataset', 'Size':dataset_size}]
        dataset_info = pd.DataFrame(dataset_info)
    dataset_info['Estimated Number of Tokens'] = num_tokens_str
    st.session_state['selected_dataset_info'] = dataset_info

    return num_tokens

def calculate_training_time(k=0):
    #Let the user enter the training time manually
    training_time = st.number_input(
        'Enter the training time in hours:',
        value=1,
        key=k,
        step=10,
    )
    k += 1
    #Display the training time
    st.markdown(f"##### Training Time = :violet[{training_time} hours]")

    #Add training time to the session state
    st.session_state['training_time'] = training_time

    return training_time*3600



def main():
    st.set_page_config(layout='wide')
    st.markdown("<h1 style='text-align: center; color: Tomato;'>Compute Optimal LLMs Training</h1>", unsafe_allow_html=True)
    add_local_gpu_info()
    
    side_cols, main_cols = st.columns([1,3])

    with main_cols:
        st.markdown("<h2 style='text-align: center; color: DodgerBlue;'>Main Computations</h2>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1,1])

        with left_col:
            with st.expander('Click to see the model parameters', expanded=False):
                P = calcualte_model_parameters(k=70)
            with st.expander('Click to see the training time', expanded=False):
                T = calculate_training_time(k=90)

        with right_col:
            with st.expander('Click to see the tao for GPU precision', expanded=False):
                tao = calculate_tao_for_gpu_precision(k=60)
            with st.expander('Click to see the dataset number of tokens', expanded=False):
                D = calculate_dataset_size(k=80)

        #check if D >= 20P
        if D >= 20*P:
            st.success(f" The dataset size is sufficient for the model parameters (D >= 20P).")
        else:
            st.warning(f" The dataset size is not sufficient for the model parameters (D < 20P).")

        st.header('Calculate Cost Using Scaling Law')
        with st.expander('Click to see the cost using scaling law', expanded=True):
            st.markdown("###### Now we can calculate the cost using the scaling law equation:")
            st.latex(r'''
                C = tao*T = 6*P*D
            ''')
            st.markdown(f"###### Where:")
            st.markdown(f"###### :blue[C is the Total Compute Required in total floating point operations")
            st.markdown(f"###### :blue[tao is the Aggregated Throughput] = :violet[**{tao} TFLOPS**]")
            st.markdown(f"###### :blue[T is the Training Time] = :violet[**{T} Second**]")
            st.markdown(f"###### :blue[P is the Number of Parameters] = :violet[**{P} Parameter**]")
            st.markdown(f"###### :blue[D is the Number of Tokens] = :violet[**{D} Tokne**]")
            st.markdown(f"###### :blue[6 is the Scaling Law Constant**]")
            st.markdown(f"---")
            
            lhs_col, rhs_col = st.columns([1,1])
            with lhs_col:
                st.markdown(f"The FLOPs based on the Hardware and Training Time")
                lhs_cost = (tao*(1000**4))*T
                st.markdown(f"###### :blue[C = tao*T = {lhs_cost} TFLOPS]")
            with rhs_col:
                st.markdown(f"The FLOPs based on the Model and Dataset Size")
                rhs_cost = 6*P*D
                st.markdown(f"###### :violet[C = 6*P*D =  {rhs_cost} TFLOPS]")

            if lhs_cost > rhs_cost:
                st.markdown(f"###### :red[The FLOPs based on the Hardware and Training Time is :blue[**higher**] than the FLOPs based on the Model and Dataset Size]")
                overtrain = lhs_cost - rhs_cost
                optimal_hours = (rhs_cost/(tao*1000**4))/3600
                optimal_throughput = (rhs_cost/T)/10**12
                st.markdown(f"###### Overtraining = :red[**{overtrain} TFLOPS**]")
                st.markdown(f"You can reduce either the training time or Hardware throughput to reduce the monetary cost.")
                st.markdown(f"###### Optimal Throughput = :red[**{optimal_throughput} TFLOPS**]")
                st.markdown(f"###### Optimal Training Time = :red[**{optimal_hours} hour(s)**]")
            else:
                st.markdown(f"###### :The FlOPs based on the Hardware and Training Time is :red[**lower**] than the FLOPs based on the Model and Dataset Size")
                undertrain = rhs_cost - lhs_cost
                optimal_hours = (rhs_cost/tao)/3600
                
                st.markdown(f"###### Undertraining = :green[**{undertrain} TFLOPS**]")
                st.markdown(f"To train the model with the parameters, dataset size, and hardware you have selected, you need to train for at least :green[**{optimal_hours :.2f} hour(s)**]")


    #Display the selected devices info and the selected precision from session state in the side_col
    with side_cols:
        st.markdown("<h2 style='text-align: center; color: DodgerBlue;'>Current Selections</h2>", unsafe_allow_html=True)
        if 'selected_devices_info' in st.session_state:
            selected_devices_info = st.session_state['selected_devices_info']
            selected_precision = st.session_state['selected_precision']
            selected_devices_info_df = pd.DataFrame(selected_devices_info)
            selected_devices_info_df['Precision'] = selected_precision
            st.markdown("###### Selected Devices Info:")
            st.write(selected_devices_info_df)
        if 'selected_dataset_info' in st.session_state:
            selected_dataset_info = st.session_state['selected_dataset_info']
            st.markdown("###### Selected Dataset Info:")
            st.write(selected_dataset_info)
        if 'selected_model_info' in st.session_state:
            selected_model_info = st.session_state['selected_model_info']
            st.markdown("###### Selected Model Info:")
            st.write(selected_model_info)
        if 'training_time' in st.session_state:
            training_time = st.session_state['training_time']
            st.markdown("###### Selected Training Time:")
            st.markdown(f'You specified **:violet[{training_time} hour(s)]**  for training.')

        st.markdown(f"---")
        st.markdown("<h2 style='text-align: center; color: DodgerBlue;'>Bibliography</h2>", unsafe_allow_html=True)
        #Citations in markdown
        st.markdown(f"###### [1] :blue[Training Compute-Optimal Large Language Models] by :blue[Jordan Hoffmann, Sebastian Borgeaud, Et. Al.]")
        st.markdown(f"###### [2] :blue[WHEN SCALING MEETS LLM FINETUNING: THE EFFECT OF DATA, MODEL AND FINETUNING METHOD] by :blue[Anonymous authors Paper under double-blind review]")
        st.markdown(f"###### [3] :blue[-] by :blue[-]")



    
    st.markdown(f"---")
    st.markdown(f"---")
    st.markdown("<h1 style='text-align: center; color: Tomato;'>Compute Optimal LLMs Fine Tuning</h1>", unsafe_allow_html=True)
    st.markdown("**We use the following multiplicative joint scaling law for LLM finetuning:**")
    st.latex(r'''
                L(X, D) = A \left( \frac{1}{X^\alpha} \right) \left( \frac{1}{D_f^\beta} \right) + E
            ''')
    st.markdown('''
                **Where:**

                    - L(X,D) represents the loss function, which quantifies the difference between the predicted output and the desired output.
                
                    - X and D are vectors of scaling factors and the fine-tuning data size, respectively. The scaling factors represent the magnitudes of the resources used during training, such as the amount of compute or the number of training examples.
                
                    - A, E, α, and β are the parameters of the model. A represents the base loss, E represents the constant error, α and β represent the exponents of the scaling factors.
                
                ''')
    

    
    st.markdown("<h2 style='text-align: center; color: DodgerBlue;'>Parameters</h3>", unsafe_allow_html=True)
    st.markdown("The paramters for :blue[**Code Generation** Task] and :orange[**LoRa** Fine-tuning] are:")
    col_g, col_m, col_p, col_t = st.columns([1, 1,1,1])
    with col_g:
        st.markdown(f"##### General Parameters:")
        st.markdown(f"###### :blue[E = 0.62]")
        st.markdown(f"###### :blue[β = 0.0.081]")
    with col_m:
        st.markdown(f"##### Scaling For Model Size:")
        st.markdown(f"###### :blue[A_m = 2.1x10^3]")
        st.markdown(f"###### :blue[α_m = 0.36]")
    with col_p:
        st.markdown(f"##### Scaling For Dataset Size:")
        st.markdown(f"###### :blue[A_p = 1.4x10^2]")
        st.markdown(f"###### :blue[α_p = 0.18]")
    with col_t:
        st.markdown(f"##### Scaling For LoRa Parameters:")
        st.markdown(f"###### :orange[A_t = 1.4]")
        st.markdown(f"###### :orange[α_t = -0.0017]")


    #Setting Variables 
    L = 0.0
    A = 0.0
    alpha = 0.0
    A_m = 2.1*10**3
    A_p = 1.4*10**2
    A_t = 1.4
    E = 0.62
    alpha_m = 0.36
    alpha_p = 0.18
    alpha_t = -0.0017
    beta = 0.081
    range_start_D = 10**6
    range_end_D = 10**10
    range_step_D = 5*10**6
    range_start_X = 10**6
    range_end_X = 10**10
    range_step_X = 5*10**6

    params_dict = {
        'A_m': A_m,
        'A_p': A_p,
        'A_t': A_t,
        'E': E,
        'alpha_m': alpha_m,
        'alpha_p': alpha_p,
        'alpha_t': alpha_t,
        'beta': beta,
        'L': L,
        'A': A,
        'alpha': alpha,
        'range_start_D': range_start_D,
        'range_end_D': range_end_D,
        'range_step_D': range_step_D,
        'range_start_X': range_start_X,
        'range_end_X': range_end_X,
        'range_step_X': range_step_X,

    }

    scaling_factors = ['Model Size', 'Dataset Size', 'LoRa Parameters']

    #Let the user select the scaling factors
    selected_scaling_factor = st.selectbox(
        'Select a scaling factor:',
        scaling_factors,
        index=0,
        key='selected_scaling_factor'
    )
    params_dict['selected_scaling_factor'] = selected_scaling_factor

    if selected_scaling_factor == 'Model Size':
        with st.expander('Click to see the scaling factor parameters', expanded=False):
            params_dict = finetuning_loss_model_size(params_dict)

    elif selected_scaling_factor == 'Dataset Size':
        with st.expander('Click to see the scaling factor parameters', expanded=False):
            params_dict = finetuning_loss_dataset_size(params_dict)

    elif selected_scaling_factor == 'LoRa Parameters':
        with st.expander('Click to see the scaling factor parameters', expanded=False):
            params_dict = finetuning_loss_lora_paramaters(params_dict)

      
    st.markdown(f'#### Based on Current Selections, Expected PPL Loss (L) = :violet[{params_dict["L"]}]')
    plot_loss_for_params(params_dict)
    with st.expander('Click to see the scaling factor parameters', expanded=False):
        st.write(params_dict)


def finetuning_loss_model_size(finetuning_scaling_params):
    # Let the user select the model size (X) and the dataset size (D)
    col_x, col_d = st.columns([1,1])
    with col_x:
        st.markdown(f"###### Model Size (X):")
        model_size = st.number_input(
            'Enter the model size:',
            value=10**9,
            key='model_size',
            step=10**6
        )
    with col_d:
        st.markdown(f"###### Dataset Size (D):")
        dataset_size = st.text_input(
            'Enter the dataset size:',
            value='100GB',
            key='dataset_size'
        )
    num_tokens = number_of_tokens_estimator(dataset_size)
    #Calculate the loss
    A = finetuning_scaling_params['A_m']
    alpha = finetuning_scaling_params['alpha_m']
    X = model_size
    D = num_tokens
    E = finetuning_scaling_params['E']
    beta = finetuning_scaling_params['beta']
    L = A*(1/X**alpha)*(1/D**beta) + E
    
    #Add the new parameters to the params_dict
    finetuning_scaling_params['X'] = X
    finetuning_scaling_params['D'] = D
    finetuning_scaling_params['L'] = L
    finetuning_scaling_params['A'] = A
    finetuning_scaling_params['alpha'] = alpha

    #Display the loss
    st.markdown(f"##### Loss (L): :violet[{L}]")

    return finetuning_scaling_params


def finetuning_loss_dataset_size(finetuning_scaling_params):
    # Let the user select the Pre-trainign dataset size (X) and the  fine tuning dataset size (D)
    col_x, col_d = st.columns([1,1])
    with col_x:
        st.markdown(f"###### Pre-training Dataset Size (X):")
        pretraining_dataset_size = st.text_input(
            'Enter the pre-training dataset size:',
            value='100GB',
            key='pretraining_dataset_size'
        )
    with col_d:
        st.markdown(f"###### Fine-tuning Dataset Size (D):")
        finetuning_dataset_size = st.text_input(
            'Enter the fine-tuning dataset size:',
            value='10GB',
            key='finetuning_dataset_size'
        )
    num_tokens_X = number_of_tokens_estimator(pretraining_dataset_size)
    num_tokens_D = number_of_tokens_estimator(finetuning_dataset_size)
    #Calculate the loss
    A = finetuning_scaling_params['A_p']
    alpha = finetuning_scaling_params['alpha_p']
    X = num_tokens_X
    D = num_tokens_D
    E = finetuning_scaling_params['E']
    beta = finetuning_scaling_params['beta']
    L = A*(1/X**alpha)*(1/D**beta) + E

    #Add the new parameters to the params_dict
    finetuning_scaling_params['X'] = X
    finetuning_scaling_params['D'] = D
    finetuning_scaling_params['L'] = L
    finetuning_scaling_params['A'] = A
    finetuning_scaling_params['alpha'] = alpha


    #Display the loss
    st.markdown(f"##### Loss (L): :violet[{L}]")

    return finetuning_scaling_params


def finetuning_loss_lora_paramaters(finetuning_scaling_params):
    # Let the user select the LoRa parameters (X) and the  fine tuning dataset size (D)
    col_x, col_d = st.columns([1,1])
    with col_x:
        st.markdown(f"###### LoRa Parameters (X):")
        lora_parameters = st.number_input(
            'Enter the LoRa parameters:',
            value=10**9,
            key='lora_parameters',
            step=10**6
        )
    with col_d:
        st.markdown(f"###### Fine-tuning Dataset Size (D):")
        finetuning_dataset_size = st.text_input(
            'Enter the fine-tuning dataset size:',
            value='10GB',
            key='finetuning_dataset_size'
        )
    num_tokens_D = number_of_tokens_estimator(finetuning_dataset_size)
    #Calculate the loss
    A = finetuning_scaling_params['A_t']
    alpha = finetuning_scaling_params['alpha_t']
    X = lora_parameters
    D = num_tokens_D
    E = finetuning_scaling_params['E']
    beta = finetuning_scaling_params['beta']
    L = A*(1/X**alpha)*(1/D**beta) + E

    #Add the new parameters to the params_dict
    finetuning_scaling_params['X'] = X
    finetuning_scaling_params['D'] = D
    finetuning_scaling_params['L'] = L
    finetuning_scaling_params['A'] = A
    finetuning_scaling_params['alpha'] = alpha

    #Display the loss
    st.markdown(f"##### Loss (L): :violet[{L}]")

    return finetuning_scaling_params


def plot_loss_for_params(finetuning_scaling_params):
    range_start_X = finetuning_scaling_params['range_start_X']
    range_end_X = finetuning_scaling_params['range_end_X']
    range_step_X = finetuning_scaling_params['range_step_X']
    range_start_D = finetuning_scaling_params['range_start_D']
    range_end_D = finetuning_scaling_params['range_end_D']
    range_step_D = finetuning_scaling_params['range_step_D']
    A = finetuning_scaling_params['A']
    alpha = finetuning_scaling_params['alpha']
    beta = finetuning_scaling_params['beta']
    E = finetuning_scaling_params['E']    

    D = finetuning_scaling_params['D']
    X_range = list(range(range_start_X, range_end_X, range_step_X))
    if len(X_range) > 100:
        X_range = X_range[::10]
    progress_bar = st.progress(0)
    bar_length = len(X_range)
    loss_change_with_X = []
    for i, val in enumerate(X_range):
            X = val
            L = A*(1/X**alpha)*(1/D**beta) + E
            loss_change_with_X.append(L)
            progress_bar.progress((i+1)/bar_length)

    X = finetuning_scaling_params['X']
    D_range = list(range(range_start_D, range_end_D, range_step_D))
    if len(D_range) > 100:
        D_range = D_range[::10]
    loss_change_with_D = []
    progress_bar = st.progress(0)
    for i, val in enumerate(D_range):
        D = val
        L = A*(1/X**alpha)*(1/D**beta) + E
        loss_change_with_D.append(L)
        progress_bar.progress((i+1)/bar_length)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_range, y=loss_change_with_X,
                        mode='lines',
                        name='X'))
    fig.add_trace(go.Scatter(x=D_range, y=loss_change_with_D,
                        mode='lines',
                        name='D'))
    fig.update_layout(
        title="Loss Change With Scaling Factor",
        xaxis_title="Scaling Factor",
        yaxis_title="Loss",
        legend_title="Legend Title",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
















