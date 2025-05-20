import warnings
import os
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from vectorize_docs import get_vector_db_retriever

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

###Create CategoryTypeChain
category_type_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a category classifier. Here are the category definitions:
     - setup: Installation, configuration, or environment setup issues
     - chains: Questions about LangChain chains, their creation, or usage
     - agents: Questions about LangChain agents, their creation, or usage
     - memory: Questions about memory components or state management
     - retrieval: Questions about document loading, vector stores, or retrieval
     - other: Any other category not listed above
     
     Task: Classify the issue into one of these categories based on the issue type and description.
     Input: {issue_text}
     Issue Type: {issue_type}
     Return a single category label."""),
    ("human", "Issue Type: {issue_type}\nIssue Description: {issue_text}")
])

category_type_chain = LLMChain(
    llm=llm,
    prompt=category_type_prompt,
    output_key="category_type"
)

###Create RetrieverChain
# Initialize the retriever
retriever = get_vector_db_retriever()

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a support assistant. Use the provided documentation to answer the question.
     If the documentation doesn't contain the answer, say "I don't have enough information to answer this question."
     If there is documentation, provide the URL to the documentation at the end of your answer like "[Source: url_path]".
     
     Documentation:
     {docs}
     
     Question: {question}
     
     Provide a clear, concise answer based on the documentation."""),
    ("human", "{question}")
])

retriever_chain = LLMChain(
    llm=llm,
    prompt=retriever_prompt,
    output_key="answer"
)

def get_relevant_docs(question):
    """Get relevant documentation for a question."""
    try:
        docs = retriever.invoke(question)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "No relevant documentation found."

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
        
        # Then, get the category
        print("\n3. Running Category Chain...")
        category_result = category_type_chain.invoke({
            "issue_text": issue_text,
            "issue_type": issue_type
        })
        category = category_result["category_type"]
        print(f"Category: {category}")
        
        # Get relevant docs and generate answer
        print("\n4. Retrieving relevant documentation...")
        docs = get_relevant_docs(issue_text)
        print("\n5. Generating answer...")
        answer_result = retriever_chain.invoke({
            "question": issue_text,
            "docs": docs
        })
        print(f"\nAnswer: {answer_result['answer']}")
        
        print("\n=== Final Classification ===")
        print(f"Issue Type: {issue_type}")
        print(f"Severity: {severity}")
        print(f"Category: {category}")
        print(f"Answer: {answer_result['answer']}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    test_chains()


