# LangChain Support Query Router

A customer engineering application that automatically triages and responds to Github issues on public repositories (defaulted to LangChain). The application classifies issues, assigns severity, and provides relevant documentation-based responses.

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
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python chains.py
```

## Project Structure

- `chains.py`: Main application logic and chain definitions
- `vectorize_docs.py`: Vector store setup for LangChain documentation
- `langchain_issues_dataset.csv`: Sample issues for testing
