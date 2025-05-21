from langsmith import Client
import json

# Initialize the LangSmith client
client = Client()

# Define your dataset name
dataset_name = "LangChain Support Triage E2E"

# Your ground truth examples
examples = [
    {
        "inputs": {
            "issue_text": "Title: Streaming issue with LiteLLM router\nDescription: When using the LiteLLM router, streaming responses are not working as expected. The responses are being buffered instead of streaming in real-time.\nIssue URL: https://github.com/langchain-ai/langchain/issues/31145",
            "issue_url": "https://github.com/langchain-ai/langchain/issues/31145"
        },
        "outputs": {
            "issue_type": "bug report",
            "severity": 2,
            "category": "chains",
            "answer": "This is a known issue with the LiteLLM router implementation. To fix this, you need to configure the streaming parameters correctly. [Source: https://python.langchain.com/docs/modules/model_io/models/streaming]"
        }
    },
    # Add your other 5 examples here in the same format
]

# Create the dataset
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="End-to-end evaluation dataset for LangChain support triage system"
)

# Upload examples to the dataset
for example in examples:
    client.create_example(
        inputs=example["inputs"],
        outputs=example["outputs"],
        dataset_id=dataset.id
    )

print(f"Created dataset '{dataset_name}' with {len(examples)} examples") 