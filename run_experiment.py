from langsmith import Client
from chains import triage_and_answer

# Initialize the LangSmith client
client = Client()

# Your test examples
test_cases = [
    {
        "issue_text": "Title: Streaming issue with LiteLLM router\nDescription: When using the LiteLLM router, streaming responses are not working as expected. The responses are being buffered instead of streaming in real-time.\nIssue URL: https://github.com/langchain-ai/langchain/issues/31145",
        "issue_url": "https://github.com/langchain-ai/langchain/issues/31145",
        "expected": {
            "issue_type": "bug report",
            "severity": 2,
            "category": "chains"
        }
    },
    # Add your other examples here
]

# Run the experiment
def run_experiment():
    # Create a new experiment
    experiment = client.create_experiment(
        name="LangChain Support Triage",
        description="Testing the triage and answer chain"
    )
    
    # Run each test case
    for i, test_case in enumerate(test_cases):
        # Run the chain
        result = triage_and_answer(
            issue_text=test_case["issue_text"],
            issue_url=test_case["issue_url"]
        )
        
        # Create a run in the experiment
        run = client.create_run(
            name=f"test_case_{i+1}",
            run_type="chain",
            inputs={
                "issue_text": test_case["issue_text"],
                "issue_url": test_case["issue_url"]
            },
            outputs=result,
            experiment_id=experiment.id
        )
        
        # Add feedback for each component
        for key in ["issue_type", "severity", "category"]:
            predicted = result.get(key, "")
            expected = test_case["expected"].get(key, "")
            score = 1.0 if str(predicted).lower() == str(expected).lower() else 0.0
            
            client.create_feedback(
                run_id=run.id,
                key=f"{key}_accuracy",
                score=score,
                comment=f"Predicted: {predicted}, Expected: {expected}"
            )

if __name__ == "__main__":
    run_experiment()
    print("Experiment complete! Check the LangSmith UI for results.") 