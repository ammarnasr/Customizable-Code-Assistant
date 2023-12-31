import os
import utils
import base64
import pickle
import joblib
import datetime
import requests
import pandas as pd
import streamlit as st
from repo_search_utils import github_scraper as gs
from repo_search_utils import search_params_st as search_params
from repo_search_utils.query_builder import GitHubRepoSearchQueryBuilder as QueryBuilder
from tqdm.auto import tqdm
from datasets import load_dataset
from huggingface_hub import login
from github import Github, Auth



GLOBAL_KEY = 0

def load_cached_searches():
    props = utils.get_app_properties()
    cache_dict_file = props['Repo Search']['cache_path']
    if os.path.exists(cache_dict_file):
        cache_dict = joblib.load(cache_dict_file)
    else:
        cache_dict = {}
    return cache_dict, cache_dict_file

def call_github_api(query, max_results, t2):
    token = gs.authenticate()
    # with st.spinner('Searching...'):
    num_pages = max_results//30
    if max_results % 30 > 0:
        num_pages += 1

    outputs = []
    for page in range(num_pages):
        repos = gs.get_repos_in_page(page, token, query)
        if repos:
            outputs.extend(repos)
        # st.spinner(f'Searching:{len(outputs)}')
        t2.markdown(f'fetching repositories: :green[{min(len(outputs), max_results)}]')
    # result = gs.search_repos(token, query, num_pages)
    outputs = outputs[:max_results]
    repos_df = gs.extract_repo_info(outputs)

    return repos_df

def cache_search_results(query, max_results, t2):
    # cache_dict, cache_dict_file = load_cached_searches()
    # if query in cache_dict:
    #     if max_results in cache_dict[query]:
    #         st.write("Already in cache")
    #         return cache_dict[query][max_results]
    df = call_github_api(query, max_results, t2)
    # if query not in cache_dict:
    #     cache_dict[query] = {}
    # cache_dict[query][max_results] = df
    # utils.dump_data(cache_dict, cache_dict_file, driver='joblib')
    return df

def get_repo_files(repo_name, branch = "master", github_token=None):
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    url = f"https://api.github.com/repos/{repo_name}/git/trees/{branch}?recursive=1"
    r = requests.get(url, headers = headers)
    if r.status_code == 200:
        res = r.json()
        files = [file["path"] for file in res["tree"]]
        return files
    
    elif branch == "master":
        files = get_repo_files(repo_name, branch = "main", github_token = github_token)
        return files

    else:
        return None


def get_repo_contents(g, repo_name):
    final_contents = []
    repo = g.get_repo(repo_name)
    contents = repo.get_contents("")
    num_dirs = 0
    num_files = 0
    for content_file in contents:
        if content_file.type == "dir":
            num_dirs += 1
        else:
            num_files += 1
    with st.empty():
        while contents:
            st.write(f':blue[{repo_name}]; Processed :green[{len(final_contents)}] file; In Queue :orange[{num_dirs}] directories and :green[{num_files}] files, total: :blue[{num_dirs + num_files}]')
            file_content = contents.pop(0)
            if file_content.type == "dir":
                num_dirs -= 1
                current_dir_contents = repo.get_contents(file_content.path)
                for c in current_dir_contents:
                    if c.type == "dir":
                        num_dirs += 1
                    else:
                        num_files += 1
                contents = current_dir_contents + contents
            else:
                num_files -= 1
                final_contents.append(file_content)
    return final_contents

def get_code_contents(file):
    try:
        raw_url = file.download_url
        raw_content = requests.get(raw_url).text
        return raw_content
    except Exception as e:
        print(f"Error getting raw content for {file.name}: {e}")
        return []
    
def github_read_file(full_name, file_path, github_token=None):
    headers = {}
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        
    url = f'https://api.github.com/repos/{full_name}/contents/{file_path}'
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    try:
      file_content = data['content']
      file_content_encoding = data.get('encoding')
      if file_content_encoding == 'base64':
          file_content = base64.b64decode(file_content).decode()
      return file_content
    except:
      pass

def display_repo_information(src_df):
    target_row = st.session_state['current_browse_row']
    row = src_df.iloc[target_row]
    repo_name = row['name']
    repo_description = row['description']
    repo_stars = row['stars']
    repo_license = row['license']
    repo_size = row['size']
    with st.expander('Main Repo Inofrmation'):
        st.markdown(f"Repo Name: :green[{repo_name}]")
        st.markdown(f"Repo Description: :orange[{repo_description}]")
        st.markdown(f"Repo Stars: :green[{repo_stars}]")
        st.markdown(f"Repo License: :orange[{repo_license}]")
        st.markdown(f"Repo Size: :green[{repo_size/1000 :.2f} MB]")
    code = row['code']
    for programing_language in code:
        if programing_language == 'Misc':
            continue
        code_files = list(code[programing_language].keys())
        code_filename = code_files[0]
        code_content = code[programing_language][code_filename]
        with st.expander(f"Code Files ({programing_language}) - {code_filename}"):
            st.code(code_content, language=programing_language.lower(), line_numbers=True)
    return 


def calc_avg_line_length(text):
    lines = text.split('\n')
    line_lengths = [len(line) for line in lines]
    return sum(line_lengths) / len(line_lengths)

def calc_max_line_length(text):
    lines = text.split('\n')
    line_lengths = [len(line) for line in lines]
    return max(line_lengths)

def calc_alphanum_fraction(text):
    if len(text) == 0:
        return 0
    alphanum = sum(c.isalnum() for c in text)
    return alphanum / len(text)

def dataset_from_df(df):
    dataset = {
        'repo_name': [],
        'repo_url': [],
        'repo_description': [],
        'repo_stars': [],
        'repo_forks': [],
        'repo_last_updated': [],
        'repo_created_at': [],
        'repo_size': [],
        'repo_license': [],
        'language': [],
        'text': [],
        'avg_line_length': [],
        'max_line_length': [],
        'alphnanum_fraction': [],
    }
    for i in tqdm(range(len(df))):
        repo = df.iloc[i]
        code = repo['code']
        for programming_language in code:
            code_files = code[programming_language]
            for code_file in code_files:
                text = code_files[code_file]
                dataset['repo_name'].append(repo['name'])
                dataset['repo_url'].append(repo['url'])
                dataset['repo_description'].append(repo['description'])
                dataset['repo_stars'].append(repo['stars'])
                dataset['repo_forks'].append(repo['forks'])
                dataset['repo_last_updated'].append(repo['last_updated'])
                dataset['repo_created_at'].append(repo['created'])
                dataset['repo_size'].append(repo['size'])
                dataset['repo_license'].append(repo['license'])
                dataset['language'].append(programming_language)
                dataset['text'].append(text)
                dataset['avg_line_length'].append(calc_avg_line_length(text))
                dataset['max_line_length'].append(calc_max_line_length(text))
                dataset['alphnanum_fraction'].append(calc_alphanum_fraction(text))
    dataset = pd.DataFrame(dataset)
    return dataset

def huggingface_dataset_from_df(df):
    dataset = dataset_from_df(df)
    with open('hf_ds.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    hf_dataset = load_dataset("pandas", data_files='hf_ds.pkl')
    os.remove('hf_ds.pkl')
    return hf_dataset





def fetch_repos_app(k=0):
    GLOBAL_KEY = k
    query = 'Query Not Built Yet'
    if 'query' not in st.session_state:
        st.session_state['query'] = query
    query_params = {}
    if 'query_params' not in st.session_state:
        st.session_state['query_params'] = query_params
    max_results = None
    if 'max_results' not in st.session_state:
        st.session_state['max_results'] = max_results
    filename = None
    if 'filename' not in st.session_state:
        st.session_state['filename'] = filename

    st.markdown('---')
    search_term = st.text_input(label='Enter The Main Search Term (Required)',
                                placeholder='For Example: "Python API"',
                                key=GLOBAL_KEY)
    GLOBAL_KEY += 1
    st.markdown(f'Entered Search Term: :green[{search_term}]')
    st.markdown('---')

    st.markdown('Note: If you want to skip a parameter, just leave it empty or with the default value')
    search_col1, search_col2, search_col3 = st.columns([1,1,1])
    with search_col1:
        with st.expander('Search_in'):
            name, description, readme, topics, GLOBAL_KEY = search_params.search_in(GLOBAL_KEY)
        with st.expander('Specific Repo'):
            owner, repo_name, GLOBAL_KEY = search_params.repo_details(GLOBAL_KEY)
        with st.expander('Specific User'):
            user, GLOBAL_KEY = search_params.user_details(GLOBAL_KEY)
   
    with search_col2:
        with st.expander('Size (1000 = 1MB)'):
            min_size, max_size, GLOBAL_KEY = search_params.repo_size_limits(GLOBAL_KEY)
        with st.expander('Forks'):
            min_forks, max_forks, GLOBAL_KEY = search_params.repo_forks_limits(GLOBAL_KEY)
        with st.expander('Stars'):
            min_stars, max_stars, GLOBAL_KEY = search_params.repo_stars_limits(GLOBAL_KEY)
        with st.expander('Created'):
            min_created, max_created, GLOBAL_KEY = search_params.repo_date_created_limits(GLOBAL_KEY)
   
    with search_col3:
        with st.expander('Language'):
            language, GLOBAL_KEY = search_params.repo_programming_language(GLOBAL_KEY)
        with st.expander('Topic'):
            topic, GLOBAL_KEY = search_params.repo_topic(GLOBAL_KEY)
        with st.expander('License'):
            license, GLOBAL_KEY = search_params.repo_license(GLOBAL_KEY)
    # st.markdown('---')

    st.markdown("Enter the max number of results to fetch (0-1000, default=100)")
    max_results = st.number_input(
        label='Max Results',
        min_value=0,
        max_value=1000,
        value=100,
        step=1,
        key=GLOBAL_KEY,
        help='Enter the max number of results to fetch (0-1000, default=100)'
    )
    GLOBAL_KEY += 1
    # st.markdown('---')

    # st.markdown("Enter the filename to save the results to (default=:orange[SearchTerm]_:blue[MaxResults].csv)")
    # # filename = st.text_input('Filename', f'{search_term}_{max_results}.csv', key=GLOBAL_KEY)
    # GLOBAL_KEY += 1
    # st.markdown('---')

    # _, center_col, _ = st.columns([6,3,6])
    # with center_col:
    #     build_query = st.button(
    #         label='Build Query',
    #         help='Press to build the query (Must Write the Search Term First)',
    #         key=GLOBAL_KEY,
    #         use_container_width=True,
    #         type='primary',
    #         disabled= search_term == ''
    #     )
    #     GLOBAL_KEY += 1

    # if build_query:
    #     qb_obj = QueryBuilder()
    #     query_args = search_params.get_query_args(name, description, readme, topics, owner, repo_name, user,
    #                                 min_size, max_size, min_forks, max_forks, min_stars, max_stars,
    #                                 min_created, max_created, language, topic, license)
    #     query_args['value'] = search_term
    #     qb_obj.init_from_args(query_args)
    #     query = qb_obj.build()
    #     st.session_state['query'] = query
    #     st.session_state['query_params'] = qb_obj.query_params
    #     st.session_state['max_results'] = max_results
    #     st.session_state['filename'] = filename

    # query, query_params, max_results, filename = st.session_state['query'], st.session_state['query_params'], st.session_state['max_results'], st.session_state['filename']
    

    st.markdown("---")
    st.markdown("#### Search Results")
    st.markdown("Press the button to search repositories")
    _, center_col, _ = st.columns([6,3,6])
    with center_col:
        search = st.button(
            label="Search",
            key=GLOBAL_KEY,
            use_container_width=True,
            type='primary',
            # disabled= build_query == False,
            help='Press to search, after query is built'
        )
    GLOBAL_KEY += 1
    if search:
        t1 = st.empty()
        t2 = st.empty()
        t3 = st.empty()
        t4 = st.empty()
        t5 = st.empty()
        t6 = st.empty()
        with st.spinner('Searching...'):
            filename = f'{search_term}_{max_results}.csv'
            #build query
            t1.markdown("building query...")
            qb_obj = QueryBuilder()
            query_args = search_params.get_query_args(name, description, readme, topics, owner, repo_name, user,
                                        min_size, max_size, min_forks, max_forks, min_stars, max_stars,
                                        min_created, max_created, language, topic, license)
            query_args['value'] = search_term
            qb_obj.init_from_args(query_args)
            query = qb_obj.build()
            st.session_state['query'] = query
            st.session_state['query_params'] = qb_obj.query_params
            st.session_state['max_results'] = max_results
            st.session_state['filename'] = filename
            query = st.session_state['query']
            query, query_params, max_results, filename = st.session_state['query'], st.session_state['query_params'], st.session_state['max_results'], st.session_state['filename']
            t1.markdown("building query finished")
            GLOBAL_KEY += 1

            df = cache_search_results(query, max_results, t2)
            # st.table(df.head(10))
            # filepath = f'./saved_searches/{filename}'
            # save_dir = './saved_searches'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # df.to_csv(filepath, index=False)
            # st.success(f"Saved to {filepath}")
            t3.download_button(
                label="Download repositories info as CSV",
                data= df.to_csv().encode('utf-8'),
                file_name= filename,
                key=GLOBAL_KEY,
                mime='text/csv',
            )
            GLOBAL_KEY += 1
            t4.markdown("extracting code from repos...")
            GLOBAL_KEY += 1
            extract_code_fun(df, filename, t5, GLOBAL_KEY)

def extract_code_fun(df, filename, t5, GLOBAL_KEY, k=0):
    code_extensions_coulmn_name_dict = {
        'py': 'Python',
        'ipynb': 'Python',
        'rmd': 'R',
        'r': 'R',
        'scala': 'Scala',
        'java': 'Java',
        'js': 'JavaScript',
        'go': 'Go',

        'c': 'C',
        'cpp': 'C++',
        'cs': 'C#',


        'html': 'HTML',
        'css': 'CSS',
        'php': 'PHP',

        'rb': 'Ruby',
        'pl': 'Perl',
        'jl': 'Julia',
        'kt': 'Kotlin',
        'swift': 'Swift',
        'vb': 'Visual Basic',
        'vba': 'Visual Basic',
        'vbnet': 'Visual Basic',
        'vb.net': 'Visual Basic',
        'ts': 'TypeScript',
        'tsx': 'TypeScript',
        'jsx': 'JavaScript',
        'tsx': 'TypeScript',
        'dart': 'Dart',
        'lua': 'Lua',
        'sh': 'Shell',
        'bash': 'Shell',
        'ps1': 'PowerShell',
        'psm1': 'PowerShell',
        'psd1': 'PowerShell',
        'ps1xml': 'PowerShell',
        'psc1': 'PowerShell',
        'psrc': 'PowerShell',
        'pp': 'Pascal',
        'pas': 'Pascal',
        'pl': 'Perl',
        'pm': 'Perl',
        't': 'Perl',
        'pod': 'Perl',
        

        'sql': 'SQL',
        'sh': 'Shell',
        'json': 'JSON',
        'xml': 'XML',
        'yml': 'YAML',
        'yaml': 'YAML',
        'md': 'Markdown',
        'txt': 'Text',
        'cfg': 'Config',
        'ini': 'Config',
        'conf': 'Config',
        'cfg': 'Config',
        'gitignore': 'Config',
        'gitattributes': 'Config',
        'gitmodules': 'Config',
        'gitkeep': 'Config',
        'gitconfig': 'Config',
        'git': 'Config',
    }
    extracted_filename = filename.replace(".csv", "_with_code.joblib")
    token = gs.authenticate()
    # auth = Auth.Token(token)
    # g = Github(auth=auth)
    repos_bar = st.progress(0, text="Extracting Code Files From Repos...")
    repos_bar_max = len(df)
    i = 0
    all_contents = []
    # with st.spinner('Extracting Code Files From Repos...'):
    for _, row in df.iterrows():
        repo_name = row['full_name']
        repo = repo_name
        # contents = get_repo_contents(g, repo)
        contents = get_repo_files(repo, github_token = token)
        code_files_dict = {}
        code_files_bar = st.progress(0, text=f"Extracting Code Files From {repo}...")
        j = 0
        code_files_bar_max = len(contents)
        for code_filepath in contents:
            # code_filename = code_file.name
            ext = code_filepath.split('.')[-1]
            if ext not in code_extensions_coulmn_name_dict:
                programing_language = 'Misc'
            else:
                programing_language = code_extensions_coulmn_name_dict[ext]
            # raw_content = get_code_contents(code_file)
            raw_content = github_read_file(repo_name, code_filepath, github_token=token)
            if programing_language not in code_files_dict:
                code_files_dict[programing_language] = {}
            code_files_dict[programing_language][code_filepath] = raw_content
            j += 1
            code_files_bar.progress(j/code_files_bar_max, text=f"Processed {j} of {code_files_bar_max} files in repo :blue[{repo}]")
        i += 1
        repos_bar.progress(i/repos_bar_max, text=f"Ectracted {i} of {repos_bar_max} repos")
        all_contents.append(code_files_dict)
    df['code'] = all_contents

    # utils.dump_data(df, f'./saved_searches/{extracted_filename}', driver='joblib')
    st.download_button(
                label="Download Data as CSV",
                data= df.to_csv().encode('utf-8'),
                file_name= filename,
                key=GLOBAL_KEY,
                mime='text/csv',
                
    )
    GLOBAL_KEY += 1
    st.success(f"Finished")


def extract_code_app(k=0):
    GLOBAL_KEY = k
    saved_searches = os.listdir("./saved_searches")
    saved_searches = [x for x in saved_searches if x.endswith(".csv")]
    filename = st.selectbox('Select File', saved_searches, key=GLOBAL_KEY)
    GLOBAL_KEY += 1
    code_extensions_coulmn_name_dict = {
        'py': 'Python',
        'ipynb': 'Python',
        'rmd': 'R',
        'r': 'R',
        'scala': 'Scala',
        'java': 'Java',
        'js': 'JavaScript',
        'go': 'Go',

        'c': 'C',
        'cpp': 'C++',
        'cs': 'C#',


        'html': 'HTML',
        'css': 'CSS',
        'php': 'PHP',

        'rb': 'Ruby',
        'pl': 'Perl',
        'jl': 'Julia',
        'kt': 'Kotlin',
        'swift': 'Swift',
        'vb': 'Visual Basic',
        'vba': 'Visual Basic',
        'vbnet': 'Visual Basic',
        'vb.net': 'Visual Basic',
        'ts': 'TypeScript',
        'tsx': 'TypeScript',
        'jsx': 'JavaScript',
        'tsx': 'TypeScript',
        'dart': 'Dart',
        'lua': 'Lua',
        'sh': 'Shell',
        'bash': 'Shell',
        'ps1': 'PowerShell',
        'psm1': 'PowerShell',
        'psd1': 'PowerShell',
        'ps1xml': 'PowerShell',
        'psc1': 'PowerShell',
        'psrc': 'PowerShell',
        'pp': 'Pascal',
        'pas': 'Pascal',
        'pl': 'Perl',
        'pm': 'Perl',
        't': 'Perl',
        'pod': 'Perl',
        

        'sql': 'SQL',
        'sh': 'Shell',
        'json': 'JSON',
        'xml': 'XML',
        'yml': 'YAML',
        'yaml': 'YAML',
        'md': 'Markdown',
        'txt': 'Text',
        'cfg': 'Config',
        'ini': 'Config',
        'conf': 'Config',
        'cfg': 'Config',
        'gitignore': 'Config',
        'gitattributes': 'Config',
        'gitmodules': 'Config',
        'gitkeep': 'Config',
        'gitconfig': 'Config',
        'git': 'Config',
    }
    extract = st.button(
        label="Extract",
        key=GLOBAL_KEY,
        use_container_width=True,
        type='primary',
        help='Press to extract code, after selecting a file'
    )

    GLOBAL_KEY += 1
    if extract:
        filepath = f'./saved_searches/{filename}'
        extracted_filename = filename.replace(".csv", "_with_code.joblib")
        if os.path.exists(f'./saved_searches/{extracted_filename}'):
            st.error(f"File already exists: {extracted_filename}")
            return
        df = pd.read_csv(filepath)
        g = gs.authenticate()
        repos_bar = st.progress(0, text="Extracting Code Files From Repos...")
        repos_bar_max = len(df)
        i = 0
        all_contents = []
        with st.spinner('Extracting Code Files From Repos...'):
            for _, row in df.iterrows():
                repo_name = row['name']
                repo_owner = row['owner']
                repo = f"{repo_owner}/{repo_name}"
                contents = get_repo_contents(g, repo)
                code_files_dict = {}
                code_files_bar = st.progress(0, text=f"Extracting Code Files From {repo}...")
                j = 0
                code_files_bar_max = len(contents)
                for code_file in contents:
                    code_filename = code_file.name
                    ext = code_filename.split('.')[-1]
                    if ext not in code_extensions_coulmn_name_dict:
                        programing_language = 'Misc'
                    else:
                        programing_language = code_extensions_coulmn_name_dict[ext]
                    raw_content = get_code_contents(code_file)
                    if programing_language not in code_files_dict:
                        code_files_dict[programing_language] = {}
                    code_files_dict[programing_language][code_filename] = raw_content
                    j += 1
                    code_files_bar.progress(j/code_files_bar_max, text=f"Processed {j} of {code_files_bar_max} files")
                i += 1
                repos_bar.progress(i/repos_bar_max, text=f"Ectracted {i} of {repos_bar_max} repos")
                all_contents.append(code_files_dict)
        df['code'] = all_contents
        utils.dump_data(df, f'./saved_searches/{extracted_filename}', driver='joblib')
        st.success(f"Saved to {extracted_filename}")


def browse_repo_info_app(k=0):
    GLOBAL_KEY = k
    saved_searches = os.listdir("./saved_searches")
    saved_searches = [x for x in saved_searches if x.endswith(".joblib")]
    filename = st.selectbox('Select File', saved_searches, key=GLOBAL_KEY)
    GLOBAL_KEY += 1
    filepath = f'./saved_searches/{filename}'
    df = joblib.load(filepath)
    orginal_df = df.copy()

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    applied_filters = []

    with filter_col1:
        
        with st.expander('Filter by Min Stars'):
            min_stars = st.number_input("Min Stars", 0, 100000, 666, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_stars != 666:
                df = df[df['stars'] >= min_stars]
                applied_filters.append(f"Min Stars: {min_stars}")

        with st.expander('Filter by License'):
            options = df['license'].unique().tolist()
            options = options + ['skip']
            license = st.selectbox('License', options, index=len(options)-1, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if license != "skip":
                df = df[df['license'] == license]   
                applied_filters.append(f"License: {license}")

        with st.expander('Filter by Language'):
            options = df['languages'].unique().tolist()
            options = [list(eval(x).keys()) for x in options]
            options = [item for sublist in options for item in sublist]
            options = list(set(options))
            options.sort()
            options = options + ['skip']
            language = st.selectbox('Language', options, index=len(options)-1, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if language != "skip":
                df = df[df['languages'].str.contains(language)]
                applied_filters.append(f"Language: {language}")

    with filter_col2:
    
        with st.expander('Filter by Topics'):
            options = df['topics'].unique().tolist()
            options = [list(eval(x)) for x in options]
            options = [item for sublist in options for item in sublist]
            options = list(set(options))
            options.sort()
            options = options + ['skip']
            topic = st.selectbox('Topic', options, index=len(options)-1, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if topic != "skip":
                df = df[df['topics'].str.contains(topic)]
                applied_filters.append(f"Topic: {topic}")

        with st.expander('Filter by Size (1000 = 1MB)'):
            min_size = st.number_input("Min Size", 0, 100000, 666, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_size != 666:
                df = df[df['size'] >= min_size]
                applied_filters.append(f"Min Size: {min_size}")

        with st.expander('Filter by Number of Commits'):
            min_commits = st.number_input("Min Commits", 0, 100000, 666, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_commits != 666:
                df = df[df['commits'] >= min_commits]
                applied_filters.append(f"Min Commits: {min_commits}")

    with filter_col3:

        with st.expander('Filter by Number of Forks'):
            min_forks = st.number_input("Min Forks", 0, 100000, 666, key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_forks != 666:
                df = df[df['forks'] >= min_forks]
                applied_filters.append(f"Min Forks: {min_forks}")

        with st.expander('Filter by Created'):
            min_created = st.date_input("Min Created", datetime.date(2000, 1, 1), key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_created != datetime.date(2000, 1, 1):
                df['created'] = pd.to_datetime(df['created']).dt.date
                df = df[df['created'] >= min_created]
                applied_filters.append(f"Min Created: {min_created}")

        with st.expander('Filter by Last Updated'):
            min_last_updated = st.date_input("Min Last Updated", datetime.date(2000, 1, 1), key=GLOBAL_KEY)
            GLOBAL_KEY += 1
            if min_last_updated != datetime.date(2000, 1, 1):
                df['last_updated'] = pd.to_datetime(df['last_updated']).dt.date
                df = df[df['last_updated'] >= min_last_updated]
                applied_filters.append(f"Min Last Updated: {min_last_updated}")
   
    st.markdown("#### After Filter")
    st.write(f"Number of Repos: is now :green[ {len(df)} ] from :red[ {len(orginal_df)} ]")
    st.write(f"Applied Filters: {applied_filters}")
    st.write(df.head())



    if 'current_browse_row' not in st.session_state:
        current_row = 0
        st.session_state['current_browse_row'] = current_row
    else:
        current_row = st.session_state['current_browse_row']

    st.markdown(f"#### Browse row by row")

    col1,_ ,col2, col3 = st.columns([2,4,12,2])

    with col1:
        prev = st.button(
            label="Prev",
            key=GLOBAL_KEY,
            use_container_width=True,
            type='primary',
            help='Press to go to the previous row',
            disabled= current_row == 0
        )
        GLOBAL_KEY += 1
        if prev:
            current_row -= 1
            if current_row == -1:
                current_row = 0
            st.session_state['current_browse_row'] = current_row
                
    with col3:
        next = st.button(
            label="Next",
            key=GLOBAL_KEY,
            use_container_width=True,
            type='primary',
            help='Press to go to the next row',
            disabled= current_row == len(df)-1
        )
        GLOBAL_KEY += 1
        if next:
            current_row += 1
            if current_row == len(df):
                current_row = len(df)-1
            st.session_state['current_browse_row'] = current_row

    with col2:
        st.write(f"curent row: {current_row} out of {len(df)-1}")

    display_repo_information(df)


def upload_dataset_hugging_face(k=0):
    GLOBAL_KEY = k
    saved_searches = os.listdir("./saved_searches")
    saved_searches = [x for x in saved_searches if x.endswith(".joblib")]
    saved_searches.append('skip')
    filename = st.selectbox('Select File', saved_searches, key=GLOBAL_KEY, index=len(saved_searches)-1)
    if filename == 'skip':
        return
    GLOBAL_KEY += 1
    filepath = f'./saved_searches/{filename}'
    df = joblib.load(filepath)
    with st.spinner("Creating Hugging Face Dataset..."):
        st.write(df.head())
        hf_dataset = huggingface_dataset_from_df(df)
    
    hf_token = st.text_input(
        label='Hugging Face Token',
        key=GLOBAL_KEY,
        help='Token can be found in your Hugging Face profile page (https://huggingface.co/settings/tokens)',
        type='password',
        value='',
    )
    GLOBAL_KEY += 1

    if hf_token != '':
        try:
            login(token=hf_token)
            st.success("Logged in successfully")
        except Exception as e:
            st.error(f"Error logging in: {e}")
            return
        
    upload_to_hub = st.button(
        label="Upload To Hub",
        key=GLOBAL_KEY,
        use_container_width=True,
        type='primary',
        help='Press to upload the dataset to Hugging Face Hub',
        disabled= hf_token == '',
    )
    GLOBAL_KEY += 1

    if upload_to_hub:
        try:
            hf_dataset.push_to_hub(f"{filename.replace('.joblib', '')}_dataset")
            st.success(f"Uploaded successfully with name: {filename.replace('.joblib', '')}_dataset")
        except Exception as e:
            st.error(f"Error uploading: {e}")
            return

