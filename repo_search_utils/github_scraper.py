import os
import json
import requests
import streamlit as st
from github import Github, Auth
import pandas as pd



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
    return access_token

def get_repos_in_page(page_num, token, query):
  headers = {
      'Accept': 'application/vnd.github+json',
      'Authorization': f'Bearer {token}',
      'X-GitHub-Api-Version': '2022-11-28',
  }

  params = {
      'q': query,
      'page' : page_num
  }

  res = requests.get('https://api.github.com/search/repositories', params=params, headers=headers)
  print(res)
  if res.status_code == 200:
    return res.json()["items"]
  else:
      return None

# def search_repos(token, query, num_pages):
#     """Search GitHub repos"""
#     outputs = []
#     for page in range(num_pages):
#         repos = get_repos_in_page(page, token, query)
#         if repos:
#             outputs.extend(repos)

    
    # return outputs

def extract_repo_info(repos_list):
    """Extract info for a single repo"""
    df = pd.DataFrame(repos_list)
    df = df[["full_name", "html_url", "description", "stargazers_count", "forks_count", "updated_at", "created_at", "owner", "topics", "size", "language", "license"]]
    return df
    
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
