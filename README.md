# LangChain Support Query Router

A customer engineering application that automatically triages and responds to Github issues on public repositories (defaulted to LangChain). The application classifies issues, assigns severity, and provides relevant documentation-based responses. It also supports eval using LangSmith.

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
python run_evaluation.py
```
- This will run the evaluation workflow using LangSmith, applying LLM-as-a-judge evaluators to all outputs.
- After completion, a link to the experiment will be printed. Open this link to view and compare results in the LangSmith UI.

### Vectorize Documentation (if needed)
```bash
python vectorize_docs.py
```
- This script builds or updates the vector store for LangChain documentation, used for retrieval in the main chain.

## Evaluation & Experiment Comparison
- Evaluations are run using `run_evaluation.py` and are tracked in LangSmith.
- All evaluators use LLM-as-a-judge for robust, context-aware scoring.

## Project Structure

- `chains.py`: Main application logic and chain definitions
- `run_evaluation.py`: Evaluation pipeline using LangSmith
- `vectorize_docs.py`: Vector store setup for LangChain documentation
- `langchain_issues_dataset.csv`: Sample issues for testing
- `.env`: Environment variables for API keys
- `requirements.txt`: Python dependencies

## Notes
- Make sure your dataset (`langchain_issues_dataset.csv`) includes both the issue description and URL for each example.
- All evaluation results and experiment comparisons are available in the LangSmith UI.
- You can add or modify evaluators in `run_evaluation.py` as needed for your use case.
