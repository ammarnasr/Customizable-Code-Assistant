# enigma_ai

enigma_ai is a python package for efficiently finetuning large language models (LLMs) like GPT-3 for code generation. 

## Features
- GitHub code scraper
    - Scrape quality code from GitHub based on parameters like stars, size, topics etc. to create a clean dataset for finetuning
    - Customizable to scrape code in specific languages, code styles etc.  
- Optimization tools
    - Find optimal hyperparameters like learning rate, batch size etc for your model and dataset to get best results
    - Supports major LLMs like GPT-3, Codex and more
    - Tunes hyperparameters based on compute constraints and desired loss function
- Easy finetuning
    - Simple wrapper around HuggingFace and Lorå to finetune LLMs on your dataset
    - Seamless integration, trains models out-of-the-box on your GPU/TPU

## Installation

```
pip install enigma_ai
```

## Usage

### Scrape GitHub
```python
from enigma_ai import GitHubScraper

scraper = GitHubScraper()
data = scraper.scrape(topics=['machine-learning'], stars=100, max_size=1000) 
```

### Optimize hyperparameters
```python
from enigma_ai import HyperparamOptimizer

optimizer = HyperparamOptimizer(model='code-davinci-002')  

params = optimizer.tune(data, max_epochs=10, target_loss=0.2)
print(params)

# Prints optimal params like LR, BS for target loss 
```

### Finetune model
```python
from enigma_ai import Finetuner 

finetuner = Finetuner(model='code-davinci-002')

finetuner.fit(data, epochs=10, batch_size=32, lr=2e-5)
finetuner.save('my_model.pkl')  
```

And more!

## Contributing

Contributions to enigma_ai are welcome...