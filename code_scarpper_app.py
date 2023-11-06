import streamlit as st
import pandas as pd
import github_scraper as gs
from query_builder import GitHubRepoSearchQueryBuilder as QueryBuilder
import datetime
import joblib
import os

st.title("GitHub Repo Search")
global_key = 0


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



    
    


search_term = st.text_input("Search term", "pentesting")


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

with st.expander('Number of Good First Issues'):
    global_key += 1
    min_good_first_issues = st.number_input("Min Good First Issues", 0, 1000, 666, key=global_key)
    global_key += 1
    max_good_first_issues = st.number_input("Max Good First Issues", 0, 1000, 666, key=global_key)
    global_key += 1
    exact_good_first_issues = st.number_input("Exact Good First Issues", 0, 1000, 666, key=global_key)

with st.expander('Number of Help Wanted Issues'):
    global_key += 1
    min_help_wanted_issues = st.number_input("Min Help Wanted Issues", 0, 1000, 666, key=global_key)
    global_key += 1
    max_help_wanted_issues = st.number_input("Max Help Wanted Issues", 0, 1000, 666, key=global_key)
    global_key += 1
    exact_help_wanted_issues = st.number_input("Exact Help Wanted Issues", 0, 1000, 666, key=global_key)

with st.expander('Show Help'):
    st.write(QueryBuilder().help())




global_key += 1
build_query = st.button("Build Query", key=global_key)

query = None
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

    st.write(qb_obj.query_params)


    query = qb_obj.build()
    st.session_state['query'] = query

st.write(query)
st.code(query, language="python", line_numbers=True)


print(query)


global_key += 1
max_results = st.number_input("Max Results", 0, 1000, 1, key=global_key)
global_key += 1
search = st.button("Search", key=global_key)

if search:
    query = st.session_state['query']
    df = cache_search_results(query, max_results)
    st.table(df)
