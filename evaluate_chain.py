from langsmith import Client, RunEvaluator
from langsmith.evaluation import EvaluationResult
from chains import triage_and_answer
import json

# Initialize the LangSmith client
client = Client()

# Define custom evaluators
class IssueTypeEvaluator(RunEvaluator):
    def evaluate_run(self, run, example):
        predicted = run.outputs.get("issue_type", "")
        expected = example.outputs.get("issue_type", "")
        score = 1.0 if predicted.lower() == expected.lower() else 0.0
        return EvaluationResult(
            key="issue_type_accuracy",
            score=score,
            comment=f"Predicted: {predicted}, Expected: {expected}"
        )

class SeverityEvaluator(RunEvaluator):
    def evaluate_run(self, run, example):
        predicted = run.outputs.get("severity", 0)
        expected = example.outputs.get("severity", 0)
        score = 1.0 if predicted == expected else 0.0
        return EvaluationResult(
            key="severity_accuracy",
            score=score,
            comment=f"Predicted: {predicted}, Expected: {expected}"
        )

class CategoryEvaluator(RunEvaluator):
    def evaluate_run(self, run, example):
        predicted = run.outputs.get("category", "")
        expected = example.outputs.get("category", "")
        score = 1.0 if predicted.lower() == expected.lower() else 0.0
        return EvaluationResult(
            key="category_accuracy",
            score=score,
            comment=f"Predicted: {predicted}, Expected: {expected}"
        )

# Create a project for this evaluation
project_name = "LangChain Support Triage Evaluation"

# Run the evaluation
def run_evaluation():
    # Get the dataset
    dataset = client.read_dataset(dataset_name="LangChain Support Triage E2E")
    
    # Run the chain on each example
    for example in client.list_examples(dataset_id=dataset.id):
        # Run the chain
        result = triage_and_answer(
            issue_text=example.inputs["issue_text"],
            issue_url=example.inputs["issue_url"]
        )
        
        # Create a run
        run = client.create_run(
            name="triage_and_answer",
            run_type="chain",
            inputs=example.inputs,
            outputs=result,
            project_name=project_name
        )
        
        # Run evaluators
        evaluators = [
            IssueTypeEvaluator(),
            SeverityEvaluator(),
            CategoryEvaluator()
        ]
        
        for evaluator in evaluators:
            evaluation = evaluator.evaluate_run(run, example)
            client.create_feedback(
                run_id=run.id,
                key=evaluation.key,
                score=evaluation.score,
                comment=evaluation.comment
            )

if __name__ == "__main__":
    run_evaluation()
    print("Evaluation complete! Check the LangSmith UI for results.") 