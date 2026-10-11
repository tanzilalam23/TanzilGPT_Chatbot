import os

import requests
import streamlit as st


@st.cache_resource
def download_private_index():
    repo = st.secrets["GITHUB_DATA_REPO"]
    branch = st.secrets.get("GITHUB_DATA_BRANCH", "main")
    token = st.secrets["GITHUB_TOKEN"]

    os.makedirs("index", exist_ok=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    files = [
        "index/faiss.index",
        "index/chunks.jsonl",
    ]

    for path in files:
        url = (
            f"https://api.github.com/repos/{repo}/contents/"
            f"{path}?ref={branch}"
        )
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        with open(path, "wb") as output:
            output.write(response.content)