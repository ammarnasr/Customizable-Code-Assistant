# Accessible Code LLMs

![](images/banner.png)

This repository contains the code and models for the Accessible Code LLMs project proposed by Enigma AI. 

## About

The goal of this project is to make large language models for code more accessible by addressing the high computational training costs and privacy concerns associated with using pretrained models. 

We aim to do this through:

- Collecting high-quality, task-specific training data to lower the amount of data needed for fine-tuning
- Exploring smaller model architectures that can achieve good performance on specialized tasks
- Providing an open-source API with options for privacy-preserving local fine-tuning
- Optimizing training methods to lower the computational training costs

## Contents

This repo contains:

- Data scraping and filtering scripts
- Notebooks for model training, evaluation, and cost analysis
- Trained models for code completion in Java, Ruby, Rust, and Swift
- An open-source API for accessing the models
- Tools and documentation for private local fine-tuning

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.3+
- Transformers 4.0+

### Installation

Clone this repository:

```
git clone https://github.com/enigmainc/accessible-code-llms.git
```

Install dependencies:

```
pip install -r requirements.txt
```

### Using the Models

- The trained models can be found in the `models` folder
- Example notebooks for using the models are in the `notebooks` folder 
- The open-source API is in the `api` folder

## Contributors

- [Ammar Khairi](https://github.com/ammarkhairi)
- [Mukhtar Mohammed](https://github.com/mukhtarmohammed) 
- [Muhammed Saeed](https://github.com/msaeed)

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more details.

## References

Relevant papers, reports, and other materials related to this project. 