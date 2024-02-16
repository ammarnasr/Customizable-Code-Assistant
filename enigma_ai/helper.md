# enigma_ai Development Guide

## Package Structure

The package is structured as follows:

```
enigma_ai/
    __init__.py
    data/
        __init__.py
        scrape.py
        process.py
    finetuning/  
        __init__.py
        peft.py
        params.py
        # or whaterver files you see need and name
    hardware/
        __init__.py
        # here I will add the code for Milestone 2 Compute Optimal Stuff
    utils.py
```

- `data/` is for the code scraping and processing functionality.
- `finetuning/` is for estimating training hyperparameters and fine-tuning models.
- `hardware/` is for compute optimal calaculations.

## Local Development

To develop and test your changes locally, follow these steps:

1. Ensure you have the latest version of the package pulled from the repository.

2. Traverse to the parent directory of the package Customizable-Code-Assistant.

3. Install the package in editable mode:

```bash
pip install -e .
```

4. Make your changes to the relevant files in the package directory.

5. To test your changes, import and use the package as you would normally:

```python
from enigma_ai import your_module

# Use your module
```

## Pushing a New Version

1. Update the version number in `setup.py`.

2. Push your changes to the main branch.

3. Build the distribution (again after traversing to the parent directory of the package Customizable-Code-Assistant):

```bash
python -m build
```

4. Upload your distribution version to PyPI (if older version is already uploaded, you may need to delete it first using `twine`):

```bash
twine upload dist/*

# If you need to delete the older version
twine upload --skip-existing dist/*
```


5. When prompted, use the project token to authenticate as follows:

```bash
Username: __token__
Password: <project token>
``` 

5. Verify that the new version is available on PyPI by :
    
```bash
pip install --upgrade enigma_ai
```

## Testing the Code Search Utility

To test the code search functionality, follow these steps:

1. Pip install the package:

```bash
pip install enigma_ai
```

2. Run the scraping script:

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

3. Run the code extraction script:
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

4. You can now inspect the `repos_with_code` dataframe to see the fetched repositories and extracted code files.


5. Finetuning
```python
from enigma_ai.finetuning.finetuneLora import main_finetune
import sys
import argparse
print(sys.path)

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script with CLI for LLM using LoRa")
    parser.add_argument("--main_path", default="/local/musaeed/Customizable-Code-Assistant/enigma_ai/JS_filesShuffled.csv", help="Main path where the experiment will be conducted")
    parser.add_argument("--experiment_name", default="Javascript", help="Name of the experiment")
    parser.add_argument("--training_data_path", default="/local/musaeed/Customizable-Code-Assistant/enigma_ai/JS_filesShuffled.csv", help="Path to the training data CSV file")
    args = parser.parse_args()
    
    main_finetune(args.main_path, args.experiment_name, args.training_data_path)

if __name__ == "__main__":
    main()

```
