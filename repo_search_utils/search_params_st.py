import streamlit as st
import datetime

def search_in(key, default_value = False):
    name = st.checkbox('Name', value=default_value, key=key)
    key += 1
    description = st.checkbox('Description', value=default_value, key=key)
    key += 1
    readme = st.checkbox('Readme', value=default_value, key=key)
    key += 1
    topics = st.checkbox('Topics', value=default_value, key=key)
    key += 1
    return name, description, readme, topics, key

def repo_details(key, deafult_value = 'skip'):
    owner = st.text_input("Owner", deafult_value, key=key)
    key += 1
    repo_name = st.text_input("Name", deafult_value, key=key)
    key += 1
    return owner, repo_name, key

def user_details(key, deafult_value = 'skip'):
    user = st.text_input("User", deafult_value, key=key)
    key += 1
    return user, key

def repo_size_limits(key, deafult_value = 666):
    min_size = st.number_input("Min Size", 0, 100000, deafult_value, key=key)
    key += 1
    max_size = st.number_input("Max Size", 0, 100000, deafult_value, key=key)
    key += 1
    return min_size, max_size,  key

def repo_forks_limits(key, deafult_value = 666):
    min_forks = st.number_input("Min Forks", 0, 100000, deafult_value, key=key)
    key += 1
    max_forks = st.number_input("Max Forks", 0, 100000, deafult_value, key=key)
    key += 1
    return min_forks, max_forks,  key

def repo_stars_limits(key, deafult_value = 666):
    min_stars = st.number_input("Min Stars", 0, 100000, deafult_value, key=key)
    key += 1
    max_stars = st.number_input("Max Stars", 0, 100000, deafult_value, key=key)
    key += 1
    return min_stars, max_stars,  key

def repo_date_created_limits(key, deafult_value = datetime.date(2000, 1, 1)):
    min_created = st.date_input("Min Created", deafult_value, key=key)
    key += 1
    max_created = st.date_input("Max Created", deafult_value, key=key)
    key += 1
    return min_created, max_created,  key

def repo_programming_language(key, default_value = 'skip'):
    language = st.text_input("Language", default_value, key=key)
    key += 1
    return language, key

def repo_license(key, default_value = 'skip'):
    license = st.text_input("License", default_value, key=key)
    key += 1
    return license, key

def repo_topic(key, default_value = 'skip'):
    topic = st.text_input("Topic", default_value, key=key)
    key += 1
    return topic, key



def get_query_args(name, description, readme, topics, owner, repo_name, user,
                   min_size, max_size, min_forks, max_forks, min_stars, max_stars,
                   min_created, max_created, language, topic, license,
                   default_text="skip", default_number=666, default_date=datetime.date(2000, 1, 1)):
        query_args = {
            'name': name,
            'description': description,
            'readme': readme,
            'topics': topics
        }
        if owner != default_text and repo_name != default_text:
            query_args['owner'] = owner
            query_args['name'] = repo_name
        if user != default_text:
            query_args['user'] = user
        if min_size != default_number:
            query_args['min_size'] = min_size
        if max_size != default_number:
            query_args['max_size'] = max_size
        if min_forks != default_number:
            query_args['min_forks'] = min_forks
        if max_forks != default_number:
            query_args['max_forks'] = max_forks
        if min_stars != default_number:
            query_args['min_stars'] = min_stars
        if max_stars != default_number:
            query_args['max_stars'] = max_stars
        if min_created != default_date:
            query_args['min_created'] = min_created
        if max_created != default_date:
            query_args['max_created'] = max_created
        if language != default_text:
            query_args['language'] = language
        if topic != default_text:
            query_args['topic'] = topic
        if license != default_text:
            query_args['license'] = license
        return query_args

