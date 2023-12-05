import os
import json
import requests
import streamlit as st
from github import Github, Auth



def set_token(token, file_path='env.json'):
    """Set GitHub access token"""
    with open(file_path, 'r+') as f:
        env = json.load(f)
        env['GITHUB_TOKEN'] = token
        f.seek(0)
        json.dump(env, f, indent=4)
        f.truncate()


def authenticate(file_path='env.json'):
    """Authenticate with GitHub API"""
    if not os.path.exists(file_path):
        #read github token from st.secrets
        access_token = st.secrets["GITHUB_TOKEN"]
    else:
        with open(file_path, 'r') as f:
            env = json.load(f)
            access_token = env['GITHUB_TOKEN']
    auth = Auth.Token(access_token)
    g = Github(auth=auth)
    return g

def search_repos(g, query):
    """Search GitHub repos"""
    result = g.search_repositories(query=query)
    return result

def extract_repo_info(repo):
    """Extract info for a single repo"""
    repo= {
        "name": repo.name,
        "url": repo.html_url,
        "description": repo.description,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "last_updated": repo.updated_at,
        "created": repo.created_at,
        "owner": repo.owner.login,
        "owner_type": repo.owner.type,
        "owner_url": repo.owner.html_url,
        "topics": repo.get_topics(),
        "languages": repo.get_languages(),
        "license": repo.license.name if repo.license else None,
        "size": repo.size,            
    }
    return repo
    
def get_repo_contents(g, repo_name, max_files=100):
    """
    Get the contents of a repository
    """
    repo = g.get_repo(repo_name)
    contents = repo.get_contents("")
    final_contents = []
    non_dir_count = 0
    dir_count = 0
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
            dir_count += 1
        else:
            final_contents.append(file_content)
            non_dir_count += 1
        if non_dir_count  == max_files:
            print(f"Processed {non_dir_count} files")
            break
    return final_contents

def filter_code_files(contents, code_extensions=(".py", ".r", ".java")):
    """
    Filter the content files to only code files
    """
    code_contents = [x for x in contents if x.name.endswith(code_extensions)]
    return code_contents

def get_code_contents(file):
    """
    Get the raw content of a code file
    """
    try:
        raw_url = file.download_url
        raw_content = requests.get(raw_url).text
        return raw_content
    except Exception as e:
        print(f"Error getting raw content for {file.name}: {e}")
        return []
