from typing import Dict, Union


Metric = float
ParamsDict = Dict[str, Union[float, Metric]]  

def estimate_finetuning_performance(
    scaling_factor: str,
    model_size: int = None,
    dataset_size: int = None,
    lora_parameters: int = None,
    finetuning_dataset_size: int = None
) -> ParamsDict:
    """Estimate finetuning loss based on scaling factor and other parameters
    
    Args:
        scaling_factor: 'Model Size', 'Dataset Size' or 'LoRa Parameters'
        model_size: Model parameter size in billions, required if scaling by model size
        dataset_size: Dataset token count, required if scaling by model size 
        lora_parameters: LoRA parameter count, required if scaling by lora parameters  
        finetuning_dataset_size: Finetuning dataset token count
        
    Returns: 
        Dict with finetuning loss and scaling parameters
    """
    
    params = _get_default_params()
    params['selected_scaling_factor'] = scaling_factor

    if scaling_factor == 'Model Size':
        if model_size is None or dataset_size is None:
            raise ValueError("Please provide model size and dataset size for this scaling factor")
        params = _scale_by_model_size(params, model_size, dataset_size)

    elif scaling_factor == 'Dataset Size':   
        if dataset_size is None or finetuning_dataset_size is None:
            raise ValueError("Please provide dataset size and finetuning dataset size for this scaling factor")
        params = _scale_by_dataset_size(params, dataset_size, finetuning_dataset_size)
        
    elif scaling_factor == 'LoRa Parameters':
        if lora_parameters is None or finetuning_dataset_size is None:
            raise ValueError("Please provide LoRA parameters and finetuning dataset size for this scaling factor")
        params = _scale_by_lora_params(params, lora_parameters, finetuning_dataset_size)
        
    return params
    

def _get_default_params() -> ParamsDict:
    return {
        'A_m': 2.1*10**3,  
        'A_p': 1.4 * 10 ** 2,
        'A_t': 1.4,
        'E': 0.62,
        'alpha_m': 0.36,
        'alpha_p': 0.18, 
        'alpha_t': -0.0017,
        'beta': 0.081,
        'L': 0.0,
        'A': 0.0,
        'alpha': 0.0
    }

def _scale_by_model_size(params: ParamsDict, X: int, D: int) -> ParamsDict:
    A = params['A_m']  
    alpha = params['alpha_m']
    E = params['E']
    beta = params['beta']
    
    L = A * (1/X**alpha) * (1/D**beta) + E
    
    params['X'] = X 
    params['D'] = D
    params['L'] = L
    params['A'] = A
    params['alpha'] = alpha
    
    return params

def _scale_by_dataset_size(params: ParamsDict, X: int, D: int) -> ParamsDict:
    A = params['A_p']
    alpha = params['alpha_p']
    E = params['E']
    beta = params['beta']
    
    L = A * (1/X**alpha) * (1/D**beta) + E
    
    params['L'] = L
    params['A'] = A
    params['alpha'] = alpha
    
    return params

def _scale_by_lora_params(params: ParamsDict, X: int, D: int) -> ParamsDict:
    A = params['A_t']
    alpha = params['alpha_t']
    E = params['E']
    beta = params['beta']
    
    L = A * (1/X**alpha) * (1/D**beta) + E
    
    params['L'] = L
    params['A'] = A
    params['alpha'] = alpha
    
    return params



# Other scaling functions    

def main():
    params = estimate_finetuning_performance(
        scaling_factor='Model Size',
        model_size=4,
        dataset_size=1e6
    )
    print(f'The Expected finetuning loss is {params["L"]}')
    return params

if __name__ == '__main__':
    main()