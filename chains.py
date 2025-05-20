import warnings
import os
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser

#ignore warnings
warnings.filterwarnings('ignore')

#read local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

#read dataset
df = pd.read_csv('ce-app/langchain_issues_dataset.csv')

"""
Issue Chain Architecture:
   ├─► IssueTypeChain -> type of issue (bug, feature request, or support question}
   │     
   │
   ├─► SeverityTypeChain -> assigns 1-4 severity score based on rubric
   │      ├ if severity ≤ 2 -> return internal escalation template
   │      └ else (3 or 4)
   │
   ├─► CategoryTypeChain -> what category the issue falls under (setup, chains, agents, memory, retrieval, other)
   │
   ├─► RetrieverChain -> gets relevant docs from vector store to answer issue
   │
   └─► AnswerChain -> outputs (category + context + query) into final self‑help JSON
"""

###Create IssueTypeChain
llm = ChatOpenAI(temperature=0, model=llm_model)

issue_type_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a triage assistant.\nTask: Classify the issue as exactly one of:\n - bug report\n - feature request\n - support question\nInput: {issue_text}\nReturn a single label."),
    ("human", "{issue_text}")
])

issue_type_chain = LLMChain(
    llm=llm,
    prompt=issue_type_prompt,
    output_key="issue_type"
)

###Create SeverityTypeChain
severity_type_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a severity classifier. Here are the severity definitions:
     Severity 1: Critical production outage or security issue (e.g. end-user inaccessible service)
     Severity 2: Major bug blocking regular usage (e.g. enterprise feature broken)
     Severity 3: Minor bug or investigation needed (e.g. analytics wrong, non-blocking)
     Severity 4: Documentation, enhancement, or informational request (no negative impact)
     
     Task: Assign a severity score between 1 and 4 based on the issue type and issue description.
     Input: {issue_text}
     Issue Type: {issue_type}
     Return a single number."""),
    ("human", "Issue Type: {issue_type}\nIssue Description: {issue_text}")
])

severity_type_chain = LLMChain(
    llm=llm,
    prompt=severity_type_prompt,
    output_key="severity_type"
)


"""
def test_chains():
    # Get the first issue from the dataset
    test_issue = df.iloc[0]
    issue_text = f"Title: {test_issue['title']}\nDescription: {test_issue['description']}"
    
    print("\n=== Testing Issue Classification Chains ===")
    print("\nInput Issue:")
    print(issue_text)
    
    try:
        # First, get the issue type
        print("\n1. Running Issue Type Chain...")
        issue_type_result = issue_type_chain.invoke({"issue_text": issue_text})
        issue_type = issue_type_result["issue_type"]
        print(f"Issue Type: {issue_type}")
        
        # Then, get the severity
        print("\n2. Running Severity Chain...")
        severity_result = severity_type_chain.invoke({
            "issue_text": issue_text,
            "issue_type": issue_type
        })
        severity = severity_result["severity_type"]
        print(f"Severity: {severity}")
        
        print("\n=== Final Classification ===")
        print(f"Issue Type: {issue_type}")
        print(f"Severity: {severity}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    test_chains()

"""
