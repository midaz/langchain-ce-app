# LangChain Support Query Router

A customer engineering application that automatically triages and responds to Github issues on public repositories (defaulted to LangChain). The application classifies issues, assigns severity, and provides relevant documentation-based responses. It also supports comprehensive evaluation using LangSmith.

## Overview

The application processes support queries through a series of chains:
1. **Issue Type Classification**: Identifies if the query is a bug report, feature request, or support question
2. **Severity Assessment**: Assigns a severity score (1-4) based on impact
3. **Category Classification**: Categorizes the query (setup, chains, agents, memory, retrieval, other)
4. **Documentation Retrieval**: Finds relevant documentation from LangChain docs
5. **Response Generation**: Provides a helpful response with documentation links

## Usage

The application processes issues from a CSV file (`langchain_issues_dataset.csv`), which is pulled from the Github API. Each issue is analyzed and classified, with support questions receiving documentation-based responses.

## Setup

1. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
GITHUB_TOKEN=your_github_token  # Optional, for higher rate limits
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Run the Main Chain (Triage and Respond)
```bash
python chains.py
```
- This will process issues from the dataset and print out the triage and response results for each issue.

### Run the Evaluation Pipeline
```bash
python eval.py
```
- This will run the comprehensive evaluation workflow using LangSmith, applying multiple LLM-as-a-judge evaluators to all outputs.
- After completion, a link to the experiment will be printed. Open this link to view and compare results in the LangSmith UI.

### Vectorize Documentation (if needed)
```bash
python vectorize_docs.py
```
- This script builds or updates the vector store for LangChain documentation, used for retrieval in the main chain.

## Evaluation System

The application includes a comprehensive evaluation system with multiple evaluators:

### Classification Evaluators
- **Issue Type Accuracy**: Evaluates the accuracy of issue type classification
- **Severity Accuracy**: Assesses the correctness of severity assignments

### Response Quality Evaluators
- **Response Action Accuracy**: Evaluates if the response correctly addresses the issue
- **Tone Appropriateness**: Assesses professionalism, empathy, clarity, and positivity
- **Response Completeness**: Evaluates technical details, explanation quality, and references
- **Technical Accuracy**: Assesses code references, documentation usage, and terminology

### Retrieval Quality Evaluators
- **Relevance Score**: Evaluates how relevant retrieved documents are to the issue
- **Coverage Score**: Assesses if retrieved documents cover all necessary information

All evaluators use LLM-as-a-judge for robust, context-aware scoring and provide detailed explanations for their assessments.

## Project Structure

- `chains.py`: Main application logic and chain definitions
- `eval.py`: Comprehensive evaluation pipeline using LangSmith
- `vectorize_docs.py`: Vector store setup for LangChain documentation
- `get_github_issues.py`: GitHub issue fetching and dataset creation
- `langchain_issues_dataset.csv`: Sample issues for testing
- `.env`: Environment variables for API keys
- `requirements.txt`: Python dependencies

## Notes
- Make sure your dataset (`langchain_issues_dataset.csv`) includes both the issue description and URL for each example.
- All evaluation results and experiment comparisons are available in the LangSmith UI.
- You can add or modify evaluators in `eval.py` as needed for your use case.
- The system uses GPT-4 for evaluation to ensure high-quality assessments.
