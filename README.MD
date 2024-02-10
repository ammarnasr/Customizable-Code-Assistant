# enigma_ai

Tools for simple and efficient training of LLMs for code generation.

## Installation

```bash
pip install enigma_ai
```

## Usage

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

Detailed usage instructions and API documentation can be found at [docs](https://enigma-ai.readthedocs.io/).

For more detailed documentation and examples, please visit the [docs](https://enigma-ai.readthedocs.io/).


## Installation

To use the Enigma AI package, follow these steps:

1. Navigate to the directory containing the Enigma AI package.
2. Install the required libraries by running the following command:

```bash
pip install .
```


## Finetuning the LLM for Code Generation

To finetune the LLM for your own code using CodeGen2, follow these steps:

1. Navigate to the directory of the LLM intelligence project.
2. Install the Enigma AI package by following the installation steps mentioned above.
3. Run the following script, replacing the placeholders with your specific paths and parameters:

```bash
python cli.py --main_path /PATH/Customizable-Code-Assistant/LLM-for-code-intelligence-Project/LLM-for-code-intelligence --experiment_name my_experiment --training_data_path JS_files.csv
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the [MIT License](LICENSE).

