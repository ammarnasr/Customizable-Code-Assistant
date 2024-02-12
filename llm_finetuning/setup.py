from setuptools import setup, find_packages
import sys
print(sys.path)
setup(
    name='llm_finetuning',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'numpy',
        'wandb',
        'pandas',
        'plotly',
        'seaborn',
        'argparse',
        'datasets',
        'jsonlines',
        'accelerate',
        'matplotlib',
        'transformers',
        'torch==1.10'
    ],
    entry_points={
        'console_scripts': [
            'llm-finetune=llm_finetuning.cli:main',
        ],
    },
)



