import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configuration
GITHUB_OWNER = "langchain-ai"
GITHUB_REPO = "langchain"
api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/issues"
headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"} if os.environ.get('GITHUB_TOKEN') else {}

# Fetch all open issues
params = {
    "state": "open",
    "per_page": 100,
    "since": (datetime.now() - timedelta(days=90)).isoformat()
}

all_issues = []
page = 1
while True:
    params['page'] = page
    response = requests.get(api_url, params=params, headers=headers)
    if response.status_code != 200 or not (issues := response.json()):
        break
    all_issues.extend(issues)
    page += 1

# Process issues into dataset
dataset = [
    {
        "issue_number": issue["number"],
        "title": issue["title"],
        "description": issue.get("body", ""),
        "labels": ", ".join(label["name"] for label in issue.get("labels", [])),
        "created_at": issue["created_at"],
        "comments_count": issue["comments"],
        "state": issue["state"]
    }
    for issue in all_issues
    if "pull_request" not in issue
]

# Save dataset
output_filename = os.path.join(os.path.dirname(__file__), f"{GITHUB_REPO}_issues_dataset.csv")
pd.DataFrame(dataset).to_csv(output_filename, index=False)
print(f"Saved {len(dataset)} issues to {output_filename}")