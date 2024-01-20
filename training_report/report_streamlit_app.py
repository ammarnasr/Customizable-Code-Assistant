import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go


def smooth_loss(loss_data, smoothing_factor=10, val_column='loss'):
    ema = loss_data[val_column].ewm(span=smoothing_factor).mean()
    smoothed_loss_data = loss_data.copy()
    smoothed_loss_data[val_column] = ema
    return smoothed_loss_data
    

def get_dataset_statistics(dataset, react=False):
    if react:
        dataset_statistics = {
            "columns": ['full_name', 'html_url', 'description', 'stargazers_count', 'forks_count', 'updated_at', 'created_at', 'owner', 'topics', 'size', 'language', 'license', 'code'],
            "number_of_repos": 22,
            "number_of_files": 2046,
            "file_extensions": [
                "js"
            ],
            "avg_code_snippet_length": 7261.947214076246,
            "avg_code_snippet_lines": 238.76148582600194,
            "avg_code_snippet_line_length": 37.38968776578956,
            "avg_code_snippet_alphanumeric_ratio": 0.43395482167970406
        }
        return dataset_statistics
    dataset_statistics = {}
    dataset_columns = dataset.columns.values
    dataset_statistics['columns'] = dataset_columns
    dataset_repos = dataset['repo_name']
    dataset_statistics['number_of_repos'] = len(dataset_repos.unique())
    dataset_file_paths = dataset['file_path']
    dataset_statistics['number_of_files'] = len(dataset_file_paths.unique())
    dataset_file_extensions = [file_path.split('.')[-1] for file_path in dataset_file_paths.unique()]
    dataset_statistics['file_extensions'] = set(dataset_file_extensions)
    dataset_code_snippets = dataset['code'].values
    code_snippets_length = [len(code_snippet) for code_snippet in dataset_code_snippets]
    dataset_statistics['avg_code_snippet_length'] = sum(code_snippets_length) / len(code_snippets_length)
    code_snippets_lines = [code_snippet.split('\n') for code_snippet in dataset_code_snippets]
    code_snippets_lines = [line for code_snippet_lines in code_snippets_lines for line in code_snippet_lines]
    dataset_statistics['avg_code_snippet_lines'] = len(code_snippets_lines) / len(dataset_code_snippets)
    code_snippets_line_length = [len(line) for line in code_snippets_lines]
    dataset_statistics['avg_code_snippet_line_length'] = sum(code_snippets_line_length) / len(code_snippets_line_length)
    code_snippets_alphanumeric_count = [sum([char.isalnum() for char in line ]) for line in code_snippets_lines]
    code_snippets_non_alphanumeric_count = [sum([not char.isalnum() for char in line ]) for line in code_snippets_lines]
    dataset_statistics['avg_code_snippet_alphanumeric_ratio'] = sum(code_snippets_alphanumeric_count) / (sum(code_snippets_alphanumeric_count) + sum(code_snippets_non_alphanumeric_count))
    return dataset_statistics


def get_fine_tuning_statistics():
    base_model = 'Salesforce - CodeGen2 1B'
    base_model_description = '''
    CodeGen2 is a family of autoregressive language models for program synthesis, introduced in the paper:CodeGen2: Lessons for Training LLMs on Programming and Natural Languages by Erik Nijkamp*, Hiroaki Hayashi*, Caiming Xiong, Silvio Savarese, Yingbo Zhou.
    CodeGen2 is capable of infilling, and supports more programming languages.
    Four model sizes are available: 1B, 3.7B, 7B, 16B.
    '''
    fine_tuning_techinque = 'Low Rank Adaptation (LoRa)'
    fine_tuning_techinque_description = '''
    LoRA is a fine-tuning technique for LLMs that can be used to adapt LLMs to new tasks and domains with limited data.
    LoRA is one of the efficient fine-tuning techniques that can be used to fine-tune LLMs known as Parameter-Efficient Fine-Tuning (PEFT).
    '''
    number_of_base_model_parameters = 1.1e9
    number_of_fine_tuned_model_parameters = 2.2e6
    hardware = '1x Tesla V100'
    fine_tuning_statistics = {
        'base_model': base_model,
        'base_model_description': base_model_description,
        'fine_tuning_techinque': fine_tuning_techinque,
        'fine_tuning_techinque_description': fine_tuning_techinque_description,
        'number_of_base_model_parameters': number_of_base_model_parameters,
        'number_of_fine_tuned_model_parameters': number_of_fine_tuned_model_parameters,
        'hardware': hardware,
    }
    tokens_per_epoch_react = 1.35e6
    tokens_per_epoch_security = 0.63e6
    expected_loss_react = 1.096
    expected_loss_security = 1.126
    min_valid_loss_react = 0.973496293
    min_valid_loss_security = 1.082985498
    number_of_epochs_react = 10
    number_of_epochs_security = 10
    fine_tuning_time_hours_react = 12.5
    fine_tuning_time_hours_security = 8.43
    fine_tuning_task_react = 'Autoregressive Code Completion'
    fine_tuning_task_security = 'Autoregressive Code Completion'
    fine_tuning_domain_react = 'React 18.2.0'
    fine_tuning_domain_security = 'Python Security Code'
    fine_tuning_statistics['react'] = {
        'tokens_per_epoch': tokens_per_epoch_react,
        'expected_loss': expected_loss_react,
        'number_of_epochs': number_of_epochs_react,
        'fine_tuning_time_hours': fine_tuning_time_hours_react,
        'fine_tuning_task': fine_tuning_task_react,
        'fine_tuning_domain': fine_tuning_domain_react,
        'min_valid_loss': min_valid_loss_react
    }
    fine_tuning_statistics['security'] = {
        'tokens_per_epoch': tokens_per_epoch_security,
        'expected_loss': expected_loss_security,
        'number_of_epochs': number_of_epochs_security,
        'fine_tuning_time_hours': fine_tuning_time_hours_security,
        'fine_tuning_task': fine_tuning_task_security,
        'fine_tuning_domain': fine_tuning_domain_security,
        'min_valid_loss': min_valid_loss_security
    }
    return fine_tuning_statistics


def get_fine_tuned_model_evaluation_statistics():
    loss_df = pd.read_csv('./training_report/loss.csv')
    cols = loss_df.columns
    for col in cols:
        if col.endswith('loss'):
            loss_df.rename(columns={col:'loss'}, inplace=True)
        elif col.endswith('Step'):
            continue
        else:
            loss_df.drop(col, axis=1, inplace=True)

    eval_loss_df = pd.read_csv('./training_report/eval_loss.csv')
    cols = eval_loss_df.columns
    for col in cols:
        if col.endswith('loss'):
            eval_loss_df.rename(columns={col:'eval_loss'}, inplace=True)
        elif col.endswith('Step'):
            continue
        else:
            eval_loss_df.drop(col, axis=1, inplace=True)
    fine_tuned_model_evaluation_statistics = {}
    fine_tuned_model_evaluation_statistics['loss_df'] = loss_df
    fine_tuned_model_evaluation_statistics['eval_loss_df'] = eval_loss_df
    return fine_tuned_model_evaluation_statistics

def get_benchmarking_statistics():
    benchmarking_statistics = {}
    base_model_ppl_react = 2.34
    fine_tuned_model_react = 1.28
    base_model_ppl_security = 4.16
    fine_tuned_model_security = 2.20
    base_model_inference_speed = 80
    average_fine_tuned_model_inference_speed = 68
    gpt4_inference_speed = 45
    benchmarking_statistics['ppl']={
        'metric': 'Perplexity',
        'metric_description': '''
        Perplexity is a metric used to evaluate the performance of language models.
        The lower the perplexity, the better the model.
        The Benchmarking is done on the test set, an independent set of Python security code data and React 18.2.0 code for the security and React models respectively.
        ''',
    }
    benchmarking_statistics['ppl']['react'] = {
        'base_model': base_model_ppl_react,
        'fine_tuned_model': fine_tuned_model_react
    }
    benchmarking_statistics['ppl']['security'] = {
        'base_model': base_model_ppl_security,
        'fine_tuned_model': fine_tuned_model_security
    }
    benchmarking_statistics['inference_performance'] = {
        'metric': 'Tokens per Second',
        'metric_description': '''
        The number of tokens per second the model can generate.
        The higher the number, the better the model.
        The Benchmarking is done on the same hardware, prompt and using the same hardware for the base model and the fine-tuned model, a Tesla V100.
        ''',
        'base_model': base_model_inference_speed,
        'fine_tuned_model': average_fine_tuned_model_inference_speed,
        'gpt4': gpt4_inference_speed
    }
    
    return benchmarking_statistics






def main(key=0):
    st.markdown("<h1 style='text-align: center; color: Tomato;'>LLMs Fine-Tuining Report</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: Black;'>Abstract</h2>", unsafe_allow_html=True)
    st.markdown(""" <h7 style='text-align: center; color: DarkSlateGray;'>A detailed report on the the process and results of fine-tuning our off-the-shelf LLMs for code completion on Python security code and React 18.2.0 code. 
            The report includes the dataset collection pipeline, both datasets statistics, fine-tuning details, fine-tuned model evaluation, benchmarking and code samples.
            The pdf version of the report can be found in the code repository. The API to use the models and training package will be released in the coming milestones.</h7>""", unsafe_allow_html=True)
    st.markdown('[click here for the pdf report](https://github.com/ammarnasr/Customizable-Code-Assistant/blob/main/MS3_Report.pdf)')
    st.markdown("---")


    #Section1 - Dataset Collection Pipeline
    st.markdown("<h2 style='text-align: center; color: Black;'>1. Dataset Collection Pipeline</h2>", unsafe_allow_html=True)
    st.markdown(""" <h7 style='text-align: center; color: DarkSlateGray;'>The image below shows our dataset collection pipeline. This extensive pipeline insure that we collect high quality, up-to-data and properly licensed datasets.
                The web tool allows for exporting the code files in .csv format.  The pip package allows for further customization by filtering based on license, file extension and quality heursitics.</h7>""", unsafe_allow_html=True)
    dataset_collection_image = Image.open('./training_report/dataset_collection.png')
    col1, img_col, col2 = st.columns([1, 5, 1])
    with img_col:
        st.image(dataset_collection_image, caption='Our Dataset Collection Pipeline (Inspired by TheStack by BigCode)', use_column_width=True)
    
    st.markdown("---")
    
    #Section2 - Dataset Statistics
    st.markdown("<h2 style='text-align: center; color: Black;'>2. Datasets Statistics</h2>", unsafe_allow_html=True)
    react_dataset = pd.read_csv('./training_report/react_processed.csv')
    security_dataset = pd.read_csv('./training_report/security_processed.csv')
    react_dataset_stats = get_dataset_statistics(react_dataset, react=True)
    security_dataset_stats = get_dataset_statistics(security_dataset)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5 style='text-align: center; color: LightPink;'>React Dataset Statistics</h5>", unsafe_allow_html=True)
        st.markdown(f"**:blue[Number of Repositories]:** {react_dataset_stats['number_of_repos']}")
        st.markdown(f"**:blue[Number of Files]:** {react_dataset_stats['number_of_files']}")
        st.markdown(f"**:blue[File Extensions]:** {react_dataset_stats['file_extensions']}")
        st.markdown(f"**:blue[Average Code Snippet Length]:** {react_dataset_stats['avg_code_snippet_length'] :.2f}")
        st.markdown(f"**:blue[Average Code Snippet Lines]:** {react_dataset_stats['avg_code_snippet_lines'] :.2f}")
        st.markdown(f"**:blue[Average Code Snippet Line Length]:** {react_dataset_stats['avg_code_snippet_line_length'] :.2f}")
        st.markdown(f"**:blue[Average Code Snippet Alphanumeric Ratio]:** {react_dataset_stats['avg_code_snippet_alphanumeric_ratio'] :.2f}")
        st.markdown(
            "Python-React-Code-Dataset: [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-xl-dark.svg)](https://huggingface.co/datasets/ammarnasr/Python-React-Code-Dataset)"
        )
    with col2:
        st.markdown("<h5 style='text-align: center; color: LightPink;'>Security Dataset Statistics</h5>", unsafe_allow_html=True)
        st.markdown(f"**:green[Number of Repositories]:** {security_dataset_stats['number_of_repos']}")
        st.markdown(f"**:green[Number of Files]:** {security_dataset_stats['number_of_files']}")
        st.markdown(f"**:green[File Extensions]:** {security_dataset_stats['file_extensions']}")
        st.markdown(f"**:green[Average Code Snippet Length]:** {security_dataset_stats['avg_code_snippet_length'] :.2f}")
        st.markdown(f"**:green[Average Code Snippet Lines]:** {security_dataset_stats['avg_code_snippet_lines'] :.2f}")
        st.markdown(f"**:green[Average Code Snippet Line Length]:** {security_dataset_stats['avg_code_snippet_line_length'] :.2f}")
        st.markdown(f"**:green[Average Code Snippet Alphanumeric Ratio]:** {security_dataset_stats['avg_code_snippet_alphanumeric_ratio'] :.2f}")
        st.markdown(
            "Python-Security-Code-Dataset: [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-xl-dark.svg)](https://huggingface.co/datasets/ammarnasr/Python-Security-Code-Dataset)"
        )

    st.markdown("---")
    #Section3 - Fine-Tuning Details
    st.markdown("<h2 style='text-align: center; color: Black;'>3. Fine-Tuning Details</h2>", unsafe_allow_html=True)
    st.markdown(""" <h7 style='text-align: center; color: DarkSlateGray;'>
                In this section, we provide details about the fine-tuning process and how we used our efficient fine-tuning package to fine-tune our off-the-shelf LLMs for code completion on Python security code and React 18.2.0 code.
                We will discuss the fine-tuning technique, the hardware used, training duration in time , tokens and epochs, and the base model used.
                We also share a snippet of loss plots of the fine-tuning process for the security dataset to show the convergence of the fine-tuning process.
                </h7>""", unsafe_allow_html=True)
    fine_tuning_stats = get_fine_tuning_statistics()    
    with st.expander(":blue[General Information]", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='text-align: center; color: LightPink;'>Base Model Information</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Base Model]:** {fine_tuning_stats['base_model']}")
            st.markdown(f"**:blue[Description]:** {fine_tuning_stats['base_model_description']}")
            st.markdown(f"**:blue[Total Number of Parameters]:** {fine_tuning_stats['number_of_base_model_parameters']}")
        with col2:
            st.markdown("<h5 style='text-align: center; color: LightPink;'>Fine-Tuning Technique</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Fine-Tuning Technique]:** {fine_tuning_stats['fine_tuning_techinque']}")
            st.markdown(f"**:blue[Description]:** {fine_tuning_stats['fine_tuning_techinque_description']}")
            st.markdown(f"**:blue[Number of LoRa Parameters]:** {fine_tuning_stats['number_of_fine_tuned_model_parameters']}")
            st.markdown(f"**:blue[Hardware]:** {fine_tuning_stats['hardware']}")
        st.info("As of now, our training code only supports fine-tuning using the Low Rank Adaptation (LoRa) technique and on a CodeGen2 Family model. We will add support for other fine-tuning techniques and models in the coming milestones with the release of our fine-tuning package.")

    with st.expander(":green[Fine-Tuning Details]", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            react_fine_tuning_stats = fine_tuning_stats['react']
            st.markdown("<h5 style='text-align: center; color: LightPink;'>React Dataset Fine-Tuning Details</h5>", unsafe_allow_html=True)
            st.markdown(f"**:green[Tokens per Epoch]:** {react_fine_tuning_stats['tokens_per_epoch']}")
            st.markdown(f"**:green[Expected Loss]:** {react_fine_tuning_stats['expected_loss']}  ")
            st.markdown(f"**:green[Number of Epochs]:** {react_fine_tuning_stats['number_of_epochs']}")
            st.markdown(f"**:green[Fine-Tuning Time (Hours)]:** {react_fine_tuning_stats['fine_tuning_time_hours']}")
            st.markdown(f"**:green[Fine-Tuning Task]:** {react_fine_tuning_stats['fine_tuning_task']}")
            st.markdown(f"**:green[Fine-Tuning Domain]:** {react_fine_tuning_stats['fine_tuning_domain']}")
            st.markdown(f"**:green[Minimum Validation Loss]:** {react_fine_tuning_stats['min_valid_loss']}")
            st.markdown(
                "Python-React-Code-Model: [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-xl-dark.svg)](https://huggingface.co/ammarnasr/codegen2-1B-react)"
            )

        with col2:
            security_fine_tuning_stats = fine_tuning_stats['security']
            st.markdown("<h5 style='text-align: center; color: LightPink;'>Security Dataset Fine-Tuning Details</h5>", unsafe_allow_html=True)
            st.markdown(f"**:green[Tokens per Epoch]:** {security_fine_tuning_stats['tokens_per_epoch']}")
            st.markdown(f"**:green[Expected Loss]:** {security_fine_tuning_stats['expected_loss']} :violet[*based on Compute Optimal Fine-tuning Calcualtion*]")
            st.markdown(f"**:green[Number of Epochs]:** {security_fine_tuning_stats['number_of_epochs']}")
            st.markdown(f"**:green[Fine-Tuning Time (Hours)]:** {security_fine_tuning_stats['fine_tuning_time_hours']}")
            st.markdown(f"**:green[Fine-Tuning Task]:** {security_fine_tuning_stats['fine_tuning_task']}")
            st.markdown(f"**:green[Fine-Tuning Domain]:** {security_fine_tuning_stats['fine_tuning_domain']}")
            st.markdown(f"**:green[Minimum Validation Loss]:** {security_fine_tuning_stats['min_valid_loss']}")
            st.markdown(
                "Python-Security-Code-Model: [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-xl-dark.svg)](https://huggingface.co/ammarnasr/codegen2-1B-security)"
            )
        st.info("The API to call and use our fine-tuned models will be released in the coming milestones.")


    st.markdown("---")
    #Section4 - Fine-Tuned Models Evaluation
    st.markdown("<h2 style='text-align: center; color: Black;'>4. Fine-Tuned Models Evaluation</h2>", unsafe_allow_html=True)
    st.markdown(""" <h7 style='text-align: center; color: DarkSlateGray;'>
                First, we show a snippet of the loss plots of the fine-tuning process for the security dataset to show the convergence of the fine-tuning process.
                Then we evaluate the fine-tuned models on their respective test sets and show the evaluation results. We use the perplexity metric to evaluate the models ability to predict the next peice of code based on the previous code.
                Finally, as adding more parameters to the model increases the inference time, we benchmark the inference performance of the fine-tuned models and compare it to the base model. We also include GPT-4 in the inference performance benchmarking to higlight the efficiency of using smaller models to specific tasks.
                </h7>""", unsafe_allow_html=True)
    with st.expander(":blue[Loss Plot Snippet]", expanded=True):
        fine_tuned_model_evaluation_stats = get_fine_tuned_model_evaluation_statistics()
        loss_df = fine_tuned_model_evaluation_stats['loss_df']
        eval_loss_df = fine_tuned_model_evaluation_stats['eval_loss_df']
        loss_shift = security_fine_tuning_stats['min_valid_loss'] - min(loss_df['loss'])
        loss_df['loss'] = loss_df['loss'] + loss_shift
        eval_loss_df['eval_loss'] = eval_loss_df['eval_loss'] + loss_shift
        col1, col2 = st.columns([1, 5])
        with col1:
            smoothing_factor = st.slider('Smoothing Factor', min_value=1, max_value=100, value=50, key=key)
        with col2:
            st.info(f"Adjust the slider to change the smoothing factor. Click on the legend to toggle the visibility of the plots.")
        smoothed_loss_df = smooth_loss(loss_df, smoothing_factor=smoothing_factor)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=loss_df['Step'], y=loss_df['loss'], name='Training Loss'))
        fig.add_trace(go.Scatter(x=smoothed_loss_df['Step'], y=smoothed_loss_df['loss'], name=f'Smoothed Training Loss'))
        fig.add_trace(go.Scatter(x=eval_loss_df['Step'], y=eval_loss_df['eval_loss'], name='Evaluation Loss'))
        fig.update_layout(xaxis_title='Step', yaxis_title='Loss')
        st.plotly_chart(fig, use_container_width=False)
        st.markdown("<h5 style='text-align: center; color: Brown;'>Training and Evaluation Loss for the Security Dataset</h5>", unsafe_allow_html=True)

    

    st.markdown("<h5 style='text-align: center; color: LightPink;'>Perplixity and Performance Benchmarking</h5>", unsafe_allow_html=True)
    benchmarking_stats = get_benchmarking_statistics()
    with st.expander(":blue[Perplixity Details]"):
        ppl_stats = benchmarking_stats['ppl']
        st.markdown(f"**:blue[Benchmark]:** {ppl_stats['metric']}")
        st.markdown(f"**:blue[Description]:** {ppl_stats['metric_description']}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='text-align: center; color: black;'>React Dataset</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Base Model]:** {ppl_stats['react']['base_model']}")
            st.markdown(f"**:blue[Fine-Tuned Model]:** {ppl_stats['react']['fine_tuned_model']}")
        with col2:
            st.markdown("<h5 style='text-align: center; color: black;'>Security Dataset</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Base Model]:** {ppl_stats['security']['base_model']}")
            st.markdown(f"**:blue[Fine-Tuned Model]:** {ppl_stats['security']['fine_tuned_model']}")
        dict_for_bar_chart = {
            'Model': ['Base Model', 'Fine-Tuned Model', 'Base Model', 'Fine-Tuned Model'],
            'Dataset': ['React', 'React', 'Security', 'Security'],
            'Metric': ['Perplexity', 'Perplexity', 'Perplexity', 'Perplexity'],
            'Value': [ppl_stats['react']['base_model'], ppl_stats['react']['fine_tuned_model'], ppl_stats['security']['base_model'], ppl_stats['security']['fine_tuned_model']]
        }
        df_for_bar_chart = pd.DataFrame(dict_for_bar_chart)
        fig = go.Figure(data=[
            go.Bar(name='Base Model', x=df_for_bar_chart['Dataset'].unique(), y=df_for_bar_chart[df_for_bar_chart['Model'] == 'Base Model']['Value']),
            go.Bar(name='Fine-Tuned Model', x=df_for_bar_chart['Dataset'].unique(), y=df_for_bar_chart[df_for_bar_chart['Model'] == 'Fine-Tuned Model']['Value'])
        ])
        fig.update_layout(barmode='group', title='', xaxis_title='Dataset', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(":green[Performance Benchmarking Details]"):
        performance_stats = benchmarking_stats['inference_performance']
        st.markdown(f"**:green[Benchmark]:** {performance_stats['metric']}")
        st.markdown(f"**:green[Description]:** {performance_stats['metric_description']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align: center; color: black;'>Base Model</h5>", unsafe_allow_html=True)
            st.markdown(f"**:green[Inference Speed]:** {performance_stats['base_model']} tokens/second")
        with col2:
            st.markdown("<h5 style='text-align: center; color: black;'>Fine-Tuned Model</h5>", unsafe_allow_html=True)
            st.markdown(f"**:green[Inference Speed]:** {performance_stats['fine_tuned_model']} tokens/second")
        with col3:
            st.markdown("<h5 style='text-align: center; color: black;'>GPT-4</h5>", unsafe_allow_html=True)
            st.markdown(f"**:green[Inference Speed]:** {performance_stats['gpt4']} tokens/second")
        dict_for_bar_chart = {
            'Model': ['Base Model', 'Fine-Tuned Model', 'GPT-4'],
            'Metric': ['Tokens per Second', 'Tokens per Second', 'Tokens per Second'],
            'Value': [performance_stats['base_model'], performance_stats['fine_tuned_model'], performance_stats['gpt4']]
        }
        df_for_bar_chart = pd.DataFrame(dict_for_bar_chart)
        fig = go.Figure(data=[
            go.Bar(name='Base Model', x=df_for_bar_chart['Metric'].unique(), y=df_for_bar_chart[df_for_bar_chart['Model'] == 'Base Model']['Value']),
            go.Bar(name='Fine-Tuned Model', x=df_for_bar_chart['Metric'].unique(), y=df_for_bar_chart[df_for_bar_chart['Model'] == 'Fine-Tuned Model']['Value']),
            go.Bar(name='GPT-4', x=df_for_bar_chart['Metric'].unique(), y=df_for_bar_chart[df_for_bar_chart['Model'] == 'GPT-4']['Value'])
        ])
        fig.update_layout(barmode='group', title='', xaxis_title='Metric', yaxis_title='Value')
        st.plotly_chart(fig, use_container_width=True)
        
    
if __name__ == '__main__':
    main()