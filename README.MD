# enigma_ai

Tools for simple and efficient training of LLMs for code generation.

## Installation

```bash
pip install enigma_ai
```

## Usage

Detailed usage instructions and API documentation will be available soon.

For more detailed documentation about the project, please refer to the SingularityNET project [here](https://deepfunding.ai/proposal/customizable-code-assistant/)

### Scraping GitHub Repositories

```python
from enigma_ai.data import scrape

# Set up your GitHub API token
github_token = 'your_github_api_token'

# Define your search query and parameters
search_term = 'pentest'
max_results = 100
filename = 'fetched_repos.csv'

# Fetch repositories matching the query
repos_df = scrape.fetch_repos(github_token, max_results, filename, search_term, min_stars=100)

# The 'repos_df' dataframe now contains information about the fetched repositories
```

### Extracting Code from Repositories

```python
from enigma_ai.data import process
import pandas as pd

# Load the previously fetched repository data
filename = 'fetched_repos.csv'
repos_df = pd.read_csv(filename)

#Limit the number of repositories to process
repos_df = repos_df.head(1)

# Extract code files from the repositories
repos_with_code = process.extract_code_from_repos(repos_df, filename, github_token)

#Print the first 1000 characters of the README.md file of the first repository
print(repos_with_code['code'].values[0]['Markdown']['README.md'][:1000])
```

### Estimating Compute Cost and Performance

```python
import enigma_ai.cost.resources as res
import enigma_ai.cost.performance as perf

gpus = [
    res.GPUTensorCoreSpec(name="A100", clock_rate_ghz=1.41, num_tensor_cores=6912),
    res.GPUTensorCoreSpec(name="V100", clock_rate_ghz=1.53, num_tensor_cores=5120),
]
gpu_specs = res.GPUSpec(name="NVIDIA", architecture="Ampere", gpus=gpus)

# Define hardware specifications
hardware = res.HardwareSpec(gpus=[gpu_specs])

# Define experiment specifications
spec = res.ExperimentSpec(model_params=1e9, dataset=1e12, hardware=hardware, precision="fp32", hours_trained=1.0)

# Calculate compute cost
compute = res.calculate_compute_cost(spec)
print(compute)

# Performance
scaling_factor = "Model Size"
model_size = 1e9  # 1 billion parameters
dataset_size = 1e12  # 1 trillion samples
scaling_params = perf.estimate_finetuning_performance(
    scaling_factor, model_size=model_size, dataset_size=dataset_size,
)
print(f'Expected Perplexity: {scaling_params["L"]}')
```


## Finetuning the LLM for Code Generation

To finetune the LLM for your own code using CodeGen2, follow these steps:

1. Navigate to the directory of the LLM intelligence project.
2. Install the Enigma AI package by following the installation steps mentioned above.
3. Run the following script, replacing the placeholders with your specific paths and parameters:

```bash
python cli.py --main_path /PATH/Customizable-Code-Assistant/LLM-for-code-intelligence-Project/LLM-for-code-intelligence --experiment_name my_experiment --training_data_path JS_files.csv
```

## License

This project is licensed under the [MIT License](LICENSE).
