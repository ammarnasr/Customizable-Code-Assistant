import streamlit as st
import pandas as pd
import github_scraper as gs
from query_builder import GitHubRepoSearchQueryBuilder as QueryBuilder
import datetime
import joblib
import os
import matplotlib.pyplot as plt

def cache_search_results(query, max_results):
    cache_dir = os.path.join(os.getcwd(), "cache")
    cach_dict_file = os.path.join(cache_dir, "cache_dict.joblib")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if os.path.exists(cach_dict_file):
        cache_dict = joblib.load(cach_dict_file)
    else:
        cache_dict = {}
    if query in cache_dict:
        if max_results in cache_dict[query]:
            st.write("Already in cache")
            return cache_dict[query][max_results]
    g = gs.authenticate()
    result = gs.search_repos(g, query)
    repos = []
    my_bar = st.progress(0)
    i = 0
    with st.spinner('Searching...'):
        for repo in result[:max_results]:
            repo = gs.extract_repo_info(repo)
            repos.append(repo)
            i += 1
            my_bar.progress(i/max_results)
    df = pd.DataFrame(repos)
    if query not in cache_dict:
        cache_dict[query] = {}
    cache_dict[query][max_results] = df
    joblib.dump(cache_dict, cach_dict_file)
    return df


st.title("GitHub Repo Search")
global_key = 0

serch_tab, extract_tab, browse_tab = st.tabs(['Search', 'Extract', 'Browse'])

with serch_tab:
    st.markdown("## Search GitHub Repos")
    st.markdown("---")
    search_term = st.text_input("Search term", "pentesting")

    search_col1, search_col2, search_col3, search_col4 = st.columns([1,1,1,1])


    with search_col1:
            
        with st.expander('Search_in'):
            global_key += 1
            name = st.checkbox('Name', value=False, key=global_key)
            global_key += 1
            description = st.checkbox('Description', value=False, key=global_key)
            global_key += 1
            readme = st.checkbox('Readme', value=False, key=global_key)
            global_key += 1
            topics = st.checkbox('Topics', value=False, key=global_key)

        with st.expander('Specific Repo'):
            global_key += 1
            owner = st.text_input("Owner", "skip", key=global_key)
            global_key += 1
            repo_name = st.text_input("Name", "skip", key=global_key)

        with st.expander('Specific User'):
            global_key += 1
            user = st.text_input("User", "skip", key=global_key)

        with st.expander('Specific Org'):
            global_key += 1
            org = st.text_input("Org", "skip", key=global_key)

        with st.expander('Number of Good First Issues'):
            global_key += 1
            min_good_first_issues = st.number_input("Min Good First Issues", 0, 1000, 666, key=global_key)
            global_key += 1
            max_good_first_issues = st.number_input("Max Good First Issues", 0, 1000, 666, key=global_key)
            global_key += 1
            exact_good_first_issues = st.number_input("Exact Good First Issues", 0, 1000, 666, key=global_key)

    with search_col2:

        with st.expander('Size (1000 = 1MB)'):
            global_key += 1
            min_size = st.number_input("Min Size", 0, 100000, 666, key=global_key)
            global_key += 1
            max_size = st.number_input("Max Size", 0, 100000, 666, key=global_key)
            global_key += 1
            exact_size = st.number_input("Exact Size", 0, 100000, 666, key=global_key)

        with st.expander('Follower'):
            global_key += 1
            min_followers = st.number_input("Min Followers", 0, 100000, 666, key=global_key)
            global_key += 1
            max_followers = st.number_input("Max Followers", 0, 100000, 666, key=global_key)
            global_key += 1
            exact_followers = st.number_input("Exact Followers", 0, 100000, 666, key=global_key)

        with st.expander('Forks'):
            global_key += 1
            min_forks = st.number_input("Min Forks", 0, 100000, 666, key=global_key)
            global_key += 1
            max_forks = st.number_input("Max Forks", 0, 100000, 666, key=global_key)
            global_key += 1
            exact_forks = st.number_input("Exact Forks", 0, 100000, 666, key=global_key)

        with st.expander('Stars'):
            global_key += 1
            min_stars = st.number_input("Min Stars", 0, 100000, 666, key=global_key)
            global_key += 1
            max_stars = st.number_input("Max Stars", 0, 100000, 666, key=global_key)
            global_key += 1
            exact_stars = st.number_input("Exact Stars", 0, 100000, 666, key=global_key)

        with st.expander('Number of Help Wanted Issues'):
            global_key += 1
            min_help_wanted_issues = st.number_input("Min Help Wanted Issues", 0, 1000, 666, key=global_key)
            global_key += 1
            max_help_wanted_issues = st.number_input("Max Help Wanted Issues", 0, 1000, 666, key=global_key)
            global_key += 1
            exact_help_wanted_issues = st.number_input("Exact Help Wanted Issues", 0, 1000, 666, key=global_key)

    with search_col3:

        with st.expander('Created'):
            global_key += 1
            min_created = st.date_input("Min Created", datetime.date(2000, 1, 1), key=global_key)
            global_key += 1
            max_created = st.date_input("Max Created", datetime.date(2000, 1, 1), key=global_key)
            global_key += 1
            exact_created = st.date_input("Exact Created", datetime.date(2000, 1, 1), key=global_key)

        with st.expander('Pushed'):
            global_key += 1
            min_pushed = st.date_input("Min Pushed", datetime.date(2000, 1, 1), key=global_key)
            global_key += 1
            max_pushed = st.date_input("Max Pushed", datetime.date(2000, 1, 1), key=global_key)
            global_key += 1
            exact_pushed = st.date_input("Exact Pushed", datetime.date(2000, 1, 1), key=global_key)

        with st.expander('Language'):
            global_key += 1
            language = st.text_input("Language", "skip", key=global_key)

        with st.expander('Topic'):
            global_key += 1
            topic = st.text_input("Topic", "skip", key=global_key)

        with st.expander('Show Help'):
            st.write(QueryBuilder().help())

    with search_col4:
            
        with st.expander('Number of Topics'):
            global_key += 1
            min_topics = st.number_input("Min Topics", 0, 1000, 666, key=global_key)
            global_key += 1
            max_topics = st.number_input("Max Topics", 0, 1000, 666, key=global_key)
            global_key += 1
            exact_topics = st.number_input("Exact Topics", 0, 1000, 666, key=global_key)

        with st.expander('License'):
            global_key += 1
            license = st.text_input("License", "skip", key=global_key)

        with st.expander('Is Public or Private'):
            options = ('public', 'private', 'skip')
            global_key += 1
            public_or_private = st.selectbox('Public or Private', options, index=2, key=global_key)

        with st.expander('Is Archived'):
            options = ('true', 'false', 'skip')
            global_key += 1
            is_archived = st.selectbox('Is Archived', options, index=2, key=global_key)



    global_key += 1
    _, query_col, _ = st.columns([6,3,6])
    with query_col:
        build_query = st.button("Build Query", key=global_key, use_container_width=True, type='primary')
    query = 'Query Not Built Yet'
    if 'query' not in st.session_state:
        st.session_state['query'] = query

    if build_query:
        qb_obj = QueryBuilder()
        qb_obj = qb_obj.search_in(search_term, name=name, description=description, readme=readme, topics=topics)

        # Specific Repo
        if repo_name != "skip" and owner != "skip":
            qb_obj = qb_obj.repo(repo_name, owner)

        # Specific User
        if user != "skip":
            qb_obj = qb_obj.user(user)

        # Specific Org
        if org != "skip":
            qb_obj = qb_obj.org(org)

        # Size
        if exact_size != 666:
            qb_obj = qb_obj.size(exact_size)
        elif min_size != 666 and max_size != 666:
            qb_obj = qb_obj.size_range(min_size, max_size)
        elif min_size != 666:
            qb_obj = qb_obj.size_gt(min_size)
        elif max_size != 666:
            qb_obj = qb_obj.size_lt(max_size)

        # Followers
        if exact_followers != 666:
            qb_obj = qb_obj.followers(exact_followers)
        elif min_followers != 666 and max_followers != 666:
            qb_obj = qb_obj.followers_range(min_followers, max_followers)
        elif min_followers != 666:
            qb_obj = qb_obj.followers_gt(min_followers)
        elif max_followers != 666:
            qb_obj = qb_obj.followers_lt(max_followers)

        # Forks
        if exact_forks != 666:
            qb_obj = qb_obj.forks(exact_forks)
        elif min_forks != 666 and max_forks != 666:
            qb_obj = qb_obj.forks_range(min_forks, max_forks)
        elif min_forks != 666:
            qb_obj = qb_obj.forks_gt(min_forks)
        elif max_forks != 666:
            qb_obj = qb_obj.forks_lt(max_forks)

        # Stars
        if exact_stars != 666:
            qb_obj = qb_obj.stars(exact_stars)
        elif min_stars != 666 and max_stars != 666:
            qb_obj = qb_obj.stars_range(min_stars, max_stars)
        elif min_stars != 666:
            qb_obj = qb_obj.stars_gt(min_stars)
        elif max_stars != 666:
            qb_obj = qb_obj.stars_lt(max_stars)

        # Created
        if exact_created != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.created(exact_created)
        elif min_created != datetime.date(2000, 1, 1) and max_created != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.created_range(min_created, max_created)
        elif min_created != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.created_gt(min_created)
        elif max_created != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.created_lt(max_created)

        # Pushed
        if exact_pushed != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.pushed(exact_pushed)
        elif min_pushed != datetime.date(2000, 1, 1) and max_pushed != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.pushed_range(min_pushed, max_pushed)
        elif min_pushed != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.pushed_gt(min_pushed)
        elif max_pushed != datetime.date(2000, 1, 1):
            qb_obj = qb_obj.pushed_lt(max_pushed)

        # Language
        if language != "skip":
            qb_obj = qb_obj.language(language)

        # Topic
        if topic != "skip":
            qb_obj = qb_obj.topic(topic)

        # Number of Topics
        if exact_topics != 666:
            qb_obj = qb_obj.number_of_topics(exact_topics)
        elif min_topics != 666:
            qb_obj = qb_obj.number_of_topics_gt(min_topics)

        # License
        if license != "skip":
            qb_obj = qb_obj.license(license)

        # Public or Private
        if public_or_private != "skip":
            qb_obj = qb_obj.is_public_private(public_or_private)

        # Is Archived
        if is_archived != "skip":
            qb_obj = qb_obj.archived(is_archived)

        # Number of Good First Issues
        if exact_good_first_issues != 666:
            qb_obj = qb_obj.good_first_issues(exact_good_first_issues)

        # Number of Help Wanted Issues
        if exact_help_wanted_issues != 666:
            qb_obj = qb_obj.help_wanted_issues(exact_help_wanted_issues)

        st.markdown("#### Query Params")
        st.write(qb_obj.query_params)


        query = qb_obj.build()
        st.session_state['query'] = query

    st.markdown("#### Query")
    st.code(query, language="python", line_numbers=True)

    st.markdown("#### Max Number of Repos to Return")
    global_key += 1
    max_results = st.number_input("Max Results", 0, 1000, 1, key=global_key)

    st.markdown("#### Save Results to CSV")
    global_key += 1
    filename = st.text_input('Filename', f'{search_term}_{max_results}.csv', key=global_key)

    global_key += 1
    search = st.button("Search", key=global_key, use_container_width=True, type='primary')
    if search:
        query = st.session_state['query']
        df = cache_search_results(query, max_results)
        st.table(df.head(10))
        filepath = f'./saved_searches/{filename}'
        df.to_csv(filepath, index=False)
        st.success(f"Saved to {filepath}")


with extract_tab:
    st.markdown("## Extract Repo Contents")
    st.markdown("---")
    saved_searches = os.listdir("./saved_searches")
    global_key += 1
    filename = st.selectbox('Select File', saved_searches, key=global_key)

    options = ('.py', '.r', '.java', '.js', '.go', '.c', '.cpp', '.cs')
    code_extensions_coulmn_name_dict = {
        '.py': 'Python',
        '.r': 'R',
        '.java': 'Java',
        '.js': 'JavaScript',
        '.go': 'Go',
        '.c': 'C',
        '.cpp': 'C++',
        '.cs': 'C#',
    }
    global_key += 1
    code_extensions = st.multiselect('Code Extensions', options, default=['.py', '.r', '.java', '.js', '.go', '.c', '.cpp', '.cs'], key=global_key)



    global_key += 1
    extract = st.button("Extract", key=global_key)

    if extract:
        filepath = f'./saved_searches/{filename}'
        df = pd.read_csv(filepath)
        g = gs.authenticate()
        my_bar = st.progress(0)
        i = 0

        with st.spinner('Extracting...'):
            with st.empty():
                for row in df.iterrows():
                    repo_name = row[1]['name']
                    repo_owner = row[1]['owner']
                    repo = f"{repo_owner}/{repo_name}"
                    contents = gs.get_repo_contents(g, repo)
                    for code_ext in code_extensions:
                        col_name = code_extensions_coulmn_name_dict[code_ext]
                        code_contents = gs.filter_code_files(contents, code_extensions=(code_ext,))
                        code_files_dict = {}
                        for code_file in code_contents:
                            raw_content = gs.get_code_contents(code_file)
                            code_filename = code_file.name
                            code_files_dict[code_filename] = raw_content
                        df.at[i, col_name] = str(code_files_dict)
                    i += 1
                    my_bar.progress(i/len(df))
                    st.write(f"Extracted {i} of {len(df)}")


        st.table(df.head(10))
        new_filename = filename.replace(".csv", "_with_code.csv")
        new_filepath = f'./saved_searches/{new_filename}'
        df.to_csv(new_filepath, index=False)
        st.success(f"Saved to {new_filepath}")


with browse_tab:
    
    def display_row(src_df, target_row):
        row = src_df.iloc[target_row]
        repo_name = row['name']
        repo_url = row['url']
        repo_description = row['description']
        repo_stars = row['stars']
        repo_forks = row['forks']
        repo_last_updated = row['last_updated']
        repo_created = row['created']
        repo_owner = row['owner']
        repo_owner_type = row['owner_type']
        repo_owner_url = row['owner_url']
        repo_topics = row['topics']
        repo_languages = row['languages']
        repo_contributors = row['contributors']
        repo_commits = row['commits']
        repo_branches = row['branches']
        repo_releases = row['releases']
        repo_default_branch = row['default_branch']
        repo_watchers = row['watchers']
        repo_open_issues = row['open_issues']
        repo_license = row['license']
        repo_size = row['size']
        repo_Python = row['Python']
        repo_R = row['R']
        repo_Java = row['Java']
        repo_JavaScript = row['JavaScript']
        repo_Go = row['Go']
        repo_C = row['C']
        repo_Cpp = row['C++']
        repo_Cs = row['C#']

        status_color_dict = {
            'pending': 'orange',
            'accepted': 'green',
            'rejected': 'red',
        }
        repo_status = row['status']
        repo_status_time = row['status_time']
        status_color = status_color_dict[repo_status]

        st.markdown(f"Showing row {target_row} of {len(src_df)-1}")
        st.markdown(f"Status: :{status_color}[{repo_status.upper()}]")
        st.markdown(f"Last Updated: {repo_status_time}")

        inf_col1, inf_col3 = st.columns([2,1])

        with inf_col1:
            with st.expander('Main Repo Inofrmation'):

                st.markdown(f"**Name of the Repo** :\n{repo_name}")
                st.markdown('---')
                st.markdown(f"**URL of the Repo**  :{repo_url}")
                st.markdown('---')
                st.markdown(f"**Description**      :{repo_description}")
                st.markdown('---')
                st.markdown(f"**No. of Stars**     :{repo_stars}")
                st.markdown('---')
                st.markdown(f"**Licence**          :{repo_license}")
                st.markdown('---')
                st.markdown(f"**Created**          :{repo_created}")
                st.markdown('---')
                st.markdown(f"**Last Updated**     :{repo_last_updated}")
                st.markdown('---')
                st.markdown(f"**Languages**        :{repo_languages}")
                st.markdown('---')
                st.markdown(f"**Topics**           :{repo_topics}")
                st.markdown('---')
                st.markdown(f"**Size in MB**       :{repo_size/1000 :.2f}MB")

        with inf_col3:
            with st.expander('Extra Repo Inofrmation'):
                # st.write(f"No. of Forks: {repo_forks}")
                # st.write(f"No. of Contributors: {repo_contributors}")
                # st.write(f"No. of Commits: {repo_commits}")
                # st.write(f"No. of Branches: {repo_branches}")
                # st.write(f"No. of Releases: {repo_releases}")
                # st.write(f"No. of Watchers: {repo_watchers}")
                # st.write(f"No. of Open Issues: {repo_open_issues}")
                # st.write(f"Default Branch: {repo_default_branch}")
                st.markdown(f"**No. of Forks**      :{repo_forks}")
                st.markdown('---')
                st.markdown(f"**No. of Contributs** :{repo_contributors}")
                st.markdown('---')
                st.markdown(f"**No. of Commits**    :{repo_commits}")
                st.markdown('---')
                st.markdown(f"**No. of Branches**   :{repo_branches}")
                st.markdown('---')
                st.markdown(f"**No. of Releases**   :{repo_releases}")
                st.markdown('---')
                st.markdown(f"**No. of Watchers**   :{repo_watchers}")
                st.markdown('---')
                st.markdown(f"**No. of Open Issues**:{repo_open_issues}")
                st.markdown('---')
                st.markdown(f"**Default Branch**    :{repo_default_branch}")

        code_col1, code_col2, code_col3 = st.columns([2,18,2])

        with code_col2:
            python_dict = eval(repo_Python)
            number_of_python_files = len(python_dict)
            with st.expander(f'Python Files ({number_of_python_files})'):
                for key, value in python_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="python", line_numbers=True)

            r_dict = eval(repo_R)
            number_of_r_files = len(r_dict)
            with st.expander(f'R Files ({number_of_r_files})'):
                for key, value in r_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="r", line_numbers=True)

            java_dict = eval(repo_Java)
            number_of_java_files = len(java_dict)
            with st.expander(f'Java Files ({number_of_java_files})'):
                for key, value in java_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="java", line_numbers=True)

        with code_col2:
            js_dict = eval(repo_JavaScript)
            number_of_js_files = len(js_dict)
            with st.expander(f'JavaScript Files ({number_of_js_files})'):
                for key, value in js_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="javascript", line_numbers=True)

            go_dict = eval(repo_Go)
            number_of_go_files = len(go_dict)
            with st.expander(f'Go Files ({number_of_go_files})'):
                for key, value in go_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="go", line_numbers=True)

            c_dict = eval(repo_C)
            number_of_c_files = len(c_dict)
            with st.expander(f'C Files ({number_of_c_files})'):
                for key, value in c_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="c", line_numbers=True)

        with code_col2:
            cpp_dict = eval(repo_Cpp)
            number_of_cpp_files = len(cpp_dict)
            with st.expander(f'C++ Files ({number_of_cpp_files})'):
                for key, value in cpp_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="cpp", line_numbers=True)

            cs_dict = eval(repo_Cs)
            number_of_cs_files = len(cs_dict)
            with st.expander(f'C# Files ({number_of_cs_files})'):
                for key, value in cs_dict.items():
                    st.write(f"#### {key}")
                    st.code(value, language="csharp", line_numbers=True)

    def update_row_status(src_df, target_row, status):
        src_df.at[target_row, 'status'] = status
        src_df.at[target_row, 'status_time'] = datetime.datetime.now()
        return src_df


    st.markdown("## Browse Saved Searches Results")
    st.markdown("---")
    saved_searches = os.listdir("./saved_searches")
    saved_searches = [x for x in saved_searches if x.endswith("_with_code.csv")]
    global_key += 1
    filename = st.selectbox('Select File', saved_searches, key=global_key)
    filepath = f'./saved_searches/{filename}'
    df = pd.read_csv(filepath)
    orginal_df = df.copy()
    if 'status' not in df.columns:
        df['status'] = 'pending'
        df['status_time'] = datetime.datetime.now()

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    applied_filters = []

    with filter_col1:
        
        with st.expander('Filter by Min Stars'):
            global_key += 1
            min_stars = st.number_input("Min Stars", 0, 100000, 666, key=global_key)
            if min_stars != 666:
                df = df[df['stars'] >= min_stars]
                applied_filters.append(f"Min Stars: {min_stars}")

        with st.expander('Filter by License'):
            options = df['license'].unique().tolist()
            options = options + ['skip']
            global_key += 1
            license = st.selectbox('License', options, index=len(options)-1, key=global_key)
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
            global_key += 1
            language = st.selectbox('Language', options, index=len(options)-1, key=global_key)
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
            global_key += 1
            topic = st.selectbox('Topic', options, index=len(options)-1, key=global_key)
            if topic != "skip":
                df = df[df['topics'].str.contains(topic)]
                applied_filters.append(f"Topic: {topic}")

        with st.expander('Filter by Size (1000 = 1MB)'):
            global_key += 1
            min_size = st.number_input("Min Size", 0, 100000, 666, key=global_key)
            if min_size != 666:
                df = df[df['size'] >= min_size]
                applied_filters.append(f"Min Size: {min_size}")

        with st.expander('Filter by Number of Commits'):
            global_key += 1
            min_commits = st.number_input("Min Commits", 0, 100000, 666, key=global_key)
            if min_commits != 666:
                df = df[df['commits'] >= min_commits]
                applied_filters.append(f"Min Commits: {min_commits}")

    with filter_col3:

        with st.expander('Filter by Number of Forks'):
            global_key += 1
            min_forks = st.number_input("Min Forks", 0, 100000, 666, key=global_key)
            if min_forks != 666:
                df = df[df['forks'] >= min_forks]
                applied_filters.append(f"Min Forks: {min_forks}")

        with st.expander('Filter by Created'):
            global_key += 1
            min_created = st.date_input("Min Created", datetime.date(2000, 1, 1), key=global_key)
            if min_created != datetime.date(2000, 1, 1):
                df['created'] = pd.to_datetime(df['created']).dt.date
                df = df[df['created'] >= min_created]
                applied_filters.append(f"Min Created: {min_created}")

        with st.expander('Filter by Last Updated'):
            global_key += 1
            min_last_updated = st.date_input("Min Last Updated", datetime.date(2000, 1, 1), key=global_key)
            if min_last_updated != datetime.date(2000, 1, 1):
                df['last_updated'] = pd.to_datetime(df['last_updated']).dt.date
                df = df[df['last_updated'] >= min_last_updated]
                applied_filters.append(f"Min Last Updated: {min_last_updated}")
   
    st.markdown("#### After Filter")
    st.write(f"Number of Repos: is now :green[ {len(df)} ] from :red[ {len(orginal_df)} ]")
    st.write(f"Applied Filters: {applied_filters}")
    st.write(df.head())



    number_of_rows = len(df) - 1
    if 'current_browse_row' not in st.session_state:
        current_row = 0
        st.session_state['current_browse_row'] = current_row
    else:
        current_row = st.session_state['current_browse_row']

    st.markdown(f"#### Browse row by row ({current_row} of {number_of_rows})")
    fig, ax = plt.subplots(figsize=(10,1))
    status_counts = df['status'].value_counts()
    status_counts = status_counts.sort_index()
    st.write(status_counts)
    labels = ['Accepted', 'Pending', 'Rejected']
    colors = ['mediumseagreen', 'goldenrod', 'firebrick']
    if 'accepted' not in status_counts:
        colors = ['goldenrod', 'firebrick']
    if 'rejected' not in status_counts:
        colors = ['mediumseagreen', 'goldenrod']
    if 'pending' not in status_counts:
        colors = ['mediumseagreen', 'firebrick']
    for i, (status, count) in enumerate(status_counts.items()):
        ax.barh(0, count, left=status_counts[:i].sum(), color=colors[i], label=status)
    #hide both axes
    ax.axis('off')
    ax.legend(ncol=3, bbox_to_anchor=(0, 1),
            loc='lower left', fontsize='small')
    st.pyplot(fig)

    display_row(df, current_row)

    col1, col2, col3, col4, col5, col6, col7 = st.columns([2,1,3,1,3,1,2])

    with col1:
        global_key += 1
        prev = st.button("Prev", key=global_key, use_container_width=True)
        if prev:
            if current_row > 0:
                current_row -= 1
                st.session_state['current_browse_row'] = current_row
            else:
                st.warning("Already on the first row")
                
    with col7:
        global_key += 1
        next = st.button("Next", key=global_key, use_container_width=True)
        if next:
            if current_row < number_of_rows:
                current_row += 1
                st.session_state['current_browse_row'] = current_row
            else:
                st.warning("Already on the last row")

    with col3:
        global_key += 1
        reject = st.button("Reject", key=global_key, use_container_width=True, type='primary')
        if reject:
            df = update_row_status(df, current_row, 'rejected')
            st.table(df.head(1))
            df.to_csv(filepath, index=False)
            st.success(f"Row {current_row} Rejected and Saved to {filepath}")

    with col5:
        global_key += 1
        accpet = st.button("Accept", key=global_key, use_container_width=True, type='primary')
        if accpet:
            df = update_row_status(df, current_row, 'accepted')
            df.to_csv(filepath, index=False)
            st.success(f"Row {current_row} Accepted and Saved to {filepath}")
