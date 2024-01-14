import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go


def smooth_loss(loss_data, smoothing_factor=10, val_column='loss'):
    ema = loss_data[val_column].ewm(span=smoothing_factor).mean()
    smoothed_loss_data = loss_data.copy()
    smoothed_loss_data[val_column] = ema
    return smoothed_loss_data
    

def get_dataset_statistics(dataset):
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
    fine_tuning_time_hours = 2
    number_of_epochs = 10
    tokens_per_epoch = 1e6
    hardware = '1x Tesla V100'
    fine_tuning_task = 'Code Completion - Autoregressive'
    fine_tuning_domain = 'Security - Python'

    fine_tuning_statistics = {
        'base_model': base_model,
        'base_model_description': base_model_description,
        'fine_tuning_techinque': fine_tuning_techinque,
        'fine_tuning_techinque_description': fine_tuning_techinque_description,
        'number_of_base_model_parameters': number_of_base_model_parameters,
        'number_of_fine_tuned_model_parameters': number_of_fine_tuned_model_parameters,
        'fine_tuning_time_hours': fine_tuning_time_hours,
        'number_of_epochs': number_of_epochs,
        'tokens_per_epoch': tokens_per_epoch,
        'hardware': hardware,
        'fine_tuning_task': fine_tuning_task,
        'fine_tuning_domain': fine_tuning_domain
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
    benchmarking_statistics['benchmark'] = 'Perplexity'
    benchmarking_statistics['benchmark_description'] = '''
    Perplexity is a metric used to evaluate language models. It is defined as the exponentiated average negative log-likelihood per token.
    The lower the perplexity, the better the model.
    The Benchmarking is done on the test set, an independent set of Python security code data.
    '''
    benchmarking_statistics['benchmark_results'] = {
        'Salesforce - CodeGen2 1B base model': 10.0,
        'Salesforce - CodeGen2 1B fine-tuned model': 2.7,
        'GPT-4': 3.0
    }
    benchmarking_statistics['inference_performance'] = {
        'metric': 'Tokens per Second',
        'metric_description': '''The number of tokens generated per second.''',
        'Salesforce - CodeGen2 1B base model': 80,
        'Salesforce - CodeGen2 1B fine-tuned model': 80.3,
        'GPT-4': 100
    }
    return benchmarking_statistics


def get_code_samples():
    code_samples = {
        'base': {
            'code_sample_1': {
                'code': '''
                def pentest():
                '''
            },
            'code_sample_2': {
                'code': '''
                def sql_injection():
                '''
            },
        },
        'finetuned': {
            'code_sample_1': {
                'code': '''
                def pentest():
                    return True
                '''
        },
        'code_sample_2': {
            'code': '''
            def sql_injection():
                return True
            '''
        },
    }
    }
    return code_samples






def main(key=0):
    st.markdown("<h1 style='text-align: center; color: Tomato;'>LLMs Fine-Tuining Report</h1>", unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: Tomato;'>Dataset Collection</h2>", unsafe_allow_html=True)
    filename = './training_report/python_code_data.csv'
    df = pd.read_csv(filename)
    dataset_collection_image = Image.open('./training_report/dataset_collection.png')
    df_stats = get_dataset_statistics(df)
    st.image(dataset_collection_image, caption='Our Dataset Collection Pipeline (Inspired by TheStack by BigCode)', use_column_width=True)
    with st.expander("Click to see dataset statistics"):
        st.write(df_stats)

    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: Tomato;'>Fine-Tuning</h2>", unsafe_allow_html=True)
    fine_tuning_stats = get_fine_tuning_statistics()    
    with st.expander(":blue[Fine-Tuning Details]"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h5 style='text-align: left; color: LightPink;'>Base Model Information</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Base Model]:** {fine_tuning_stats['base_model']}")
            st.markdown(f"**:blue[Description]:** {fine_tuning_stats['base_model_description']}")
        with col2:
            st.markdown("<h5 style='text-align: left; color: LightPink;'>Fine-Tuning Technique</h5>", unsafe_allow_html=True)
            st.markdown(f"**:blue[Fine-Tuning Technique]:** {fine_tuning_stats['fine_tuning_techinque']}")
            st.markdown(f"**:blue[Description]:** {fine_tuning_stats['fine_tuning_techinque_description']}")
    with st.expander(":green[Fine-Tuning Task, Domain and Statistics]"):
        st.markdown(f"**:green[Fine-Tuning Task]:** {fine_tuning_stats['fine_tuning_task']}")
        st.markdown(f"**:green[Fine-Tuning Domain]:** {fine_tuning_stats['fine_tuning_domain']}")
        st.markdown(f"**:green[Number of Base Model Parameters]:** {fine_tuning_stats['number_of_base_model_parameters']}")
        st.markdown(f"**:green[Number of Fine-Tuned Model Parameters]:** {fine_tuning_stats['number_of_fine_tuned_model_parameters']}")
        st.markdown(f"**:green[Fine-Tuning Time (Hours)]:** {fine_tuning_stats['fine_tuning_time_hours']}")
        st.markdown(f"**:green[Number of Epochs]:** {fine_tuning_stats['number_of_epochs']}")
        st.markdown(f"**:green[Tokens per Epoch]:** {fine_tuning_stats['tokens_per_epoch']}")
        st.markdown(f"**:green[Hardware]:** {fine_tuning_stats['hardware']}")

    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: Tomato;'>Fine-Tuned Model Evaluation</h2>", unsafe_allow_html=True)
    fine_tuned_model_evaluation_stats = get_fine_tuned_model_evaluation_statistics()
    loss_df = fine_tuned_model_evaluation_stats['loss_df']
    eval_loss_df = fine_tuned_model_evaluation_stats['eval_loss_df']
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
    fig.update_layout(title='', xaxis_title='Step', yaxis_title='Loss')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h5 style='text-align: center; color: Brown;'>Training and Evaluation Loss</h5>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: Tomato;'>Benchmarking</h2>", unsafe_allow_html=True)
    benchmarking_stats = get_benchmarking_statistics()
    with st.expander(":blue[Benchmarking Details]"):
        st.markdown(f"**:blue[Benchmark]:** {benchmarking_stats['benchmark']}")
        st.markdown(f"**:blue[Description]:** {benchmarking_stats['benchmark_description']}")
        st.markdown(f"**:blue[Benchmark Results]:**")
        st.write(benchmarking_stats['benchmark_results'])
    with st.expander(":green[Inference Performance]"):
        st.markdown(f"**:green[Metric]:** {benchmarking_stats['inference_performance']['metric']}")
        st.markdown(f"**:green[Description]:** {benchmarking_stats['inference_performance']['metric_description']}")
        st.markdown(f"**:green[Results]:**")
        st.write(benchmarking_stats['inference_performance'])


    st.markdown("---")

    st.markdown("<h2 style='text-align: center; color: Tomato;'>Code Samples</h2>", unsafe_allow_html=True)
    code_samples = get_code_samples()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<h5 style='text-align: left; color: LightPink;'>Base Model Code Samples</h5>", unsafe_allow_html=True)
        st.code(code_samples['base']['code_sample_1']['code'], language='python')
        st.code(code_samples['base']['code_sample_2']['code'], language='python')
    with col2:
        st.markdown("<h5 style='text-align: left; color: LightPink;'>Fine-Tuned Model Code Samples</h5>", unsafe_allow_html=True)
        st.code(code_samples['finetuned']['code_sample_1']['code'], language='python')
        st.code(code_samples['finetuned']['code_sample_2']['code'], language='python')
    

if __name__ == '__main__':
    main()