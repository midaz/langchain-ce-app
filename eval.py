from langsmith import Client
from chains import triage_and_answer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import os
import sys
from langsmith.evaluation import EvaluationResult, RunEvaluator, EvaluationResults

# Read local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# ===== Evaluation Schemas =====
class SimpleJudgeEvaluation(BaseModel):
    score: float = Field(description="Overall score from 0-1")
    explanation: str = Field(description="Step-by-step explanation of the score")

class RetrievalQualityEvaluation(BaseModel):
    relevance_score: float = Field(description="Score from 0-1 for relevance of retrieved documents")
    coverage_score: float = Field(description="Score from 0-1 for coverage of required information")
    explanation: str = Field(description="Explanation of the scores")

class IssueTypeJudgeEvaluation(BaseModel):
    score: float = Field(description="Score from 0-1 for issue type classification accuracy")
    explanation: str = Field(description="Explanation of the score")

class SeverityJudgeEvaluation(BaseModel):
    score: float = Field(description="Score from 0-1 for severity accuracy")
    explanation: str = Field(description="Explanation of the score")

class ResponseActionJudgeEvaluation(BaseModel):
    score: float = Field(description="Score from 0-1 for response action accuracy")
    explanation: str = Field(description="Explanation of the score")

# ===== Evaluation Prompts =====
ISSUE_TYPE_JUDGE_PROMPT = """
You are evaluating the accuracy of an issue type classification for a customer support issue.

Expected issue type: {expected}
Predicted issue type: {predicted}

Score from 0-1 where:
1.0 = Perfect match
0.0 = Completely wrong

Explain your reasoning step by step.
"""

SEVERITY_JUDGE_PROMPT = """
You are evaluating the accuracy of a severity classification for a customer support issue.

Expected severity: {expected}
Predicted severity: {predicted}

Score from 0-1 where:
1.0 = Perfect match
0.0 = Completely wrong

Explain your reasoning step by step.
"""

RESPONSE_ACTION_JUDGE_PROMPT = """
You are evaluating whether the response action correctly addresses the issue.

Expected issue type: {expected_issue_type}
Predicted issue type: {predicted_issue_type}
Expected severity: {expected_severity}
Predicted severity: {predicted_severity}
Response: {answer}

Score from 0-1 where:
1.0 = Response action is fully appropriate
0.0 = Response action is completely inappropriate

Explain your reasoning step by step.
"""

TONE_JUDGE_PROMPT = """
You are evaluating the tone of a customer support response.

Consider professionalism, empathy, clarity, and positivity.
Score from 0-1 where:
1.0 = Perfect professional tone
0.0 = Unprofessional or inappropriate tone

Explain your reasoning step by step.
"""

COMPLETENESS_JUDGE_PROMPT = """
You are evaluating the completeness of a customer support response.

Consider technical details, explanation quality, next steps, and references.
Score from 0-1 where:
1.0 = Complete response with all necessary information
0.0 = Incomplete or missing critical information

Special instruction: If 'Relevant docs retrieved' is NO, automatically score 0.
Explain your reasoning step by step.
"""

TECHNICAL_ACCURACY_JUDGE_PROMPT = """
You are evaluating the technical accuracy of a customer support response.

Consider code references, documentation, and technical terminology.
Score from 0-1 where:
1.0 = Technically accurate in all aspects
0.0 = Contains technical inaccuracies

Special instruction: If 'Relevant docs retrieved' is NO, automatically score 0.

Explain your reasoning step by step.
"""

RETRIEVAL_QUALITY_JUDGE_PROMPT = """
You are evaluating the quality of document retrieval for a customer support issue.

Issue: {issue_text}
Retrieved Documents: {retrieved_docs}

Score from 0-1 for:
1. Relevance: How relevant are the retrieved documents to the issue? Consider:
   - Are the sources directly addressing the issue?
   - Do the sources contain specific, actionable information?
   - Are the sources up-to-date and appropriate for the issue type?

2. Coverage: Do the retrieved documents cover all necessary information? Consider:
   - Are all aspects of the issue addressed?
   - Are there any critical gaps in the information?
   - Are there multiple sources providing complementary information?

Note: If the answer indicates no documents were retrieved ("I don't have enough information" or "No relevant documentation found"), 
score both relevance and coverage as 0.

Explain your reasoning step by step.
"""

# ===== Classification Evaluators =====
# These evaluators assess how well the system categorizes the issue
class IssueTypeEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=IssueTypeJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        predicted = run.outputs.get("issue_type", "")
        expected = example.outputs.get("issue_type", "")
        prompt = ISSUE_TYPE_JUDGE_PROMPT.format(expected=expected, predicted=predicted) + "\n" + self.parser.get_format_instructions()
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="issue_type_accuracy",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class SeverityEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SeverityJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        predicted = run.outputs.get("severity", "")
        expected = example.outputs.get("severity", "")
        prompt = SEVERITY_JUDGE_PROMPT.format(expected=expected, predicted=predicted) + "\n" + self.parser.get_format_instructions()
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="severity_accuracy",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

# ===== Response Quality Evaluators =====
# These evaluators assess the quality and appropriateness of the system's response
class ResponseActionEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=ResponseActionJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        predicted_issue_type = run.outputs.get("issue_type", "")
        predicted_severity = run.outputs.get("severity", "")
        expected_issue_type = example.outputs.get("issue_type", "")
        expected_severity = example.outputs.get("severity", "")
        prompt = RESPONSE_ACTION_JUDGE_PROMPT.format(
            expected_issue_type=expected_issue_type,
            predicted_issue_type=predicted_issue_type,
            expected_severity=expected_severity,
            predicted_severity=predicted_severity,
            answer=answer
        ) + "\n" + self.parser.get_format_instructions()
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="response_action_accuracy",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class ToneAppropriatenessEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SimpleJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        prompt = f"""Response to evaluate: {answer}\n\n{TONE_JUDGE_PROMPT}\n{self.parser.get_format_instructions()}"""
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="tone_appropriateness",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class ResponseCompletenessEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SimpleJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        retrieved_docs = run.outputs.get("retrieved_docs", [])
        retrieval_status = "YES" if retrieved_docs else "NO"
        answer = run.outputs.get("answer", "")
        prompt = (
            f"Relevant docs retrieved: {retrieval_status}\n"
            f"Response to evaluate: {answer}\n\n"
            f"{COMPLETENESS_JUDGE_PROMPT}\n"
            f"{self.parser.get_format_instructions()}"
        )
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="response_completeness",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class TechnicalAccuracyEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=SimpleJudgeEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        retrieved_docs = run.outputs.get("retrieved_docs", [])
        retrieval_status = "YES" if retrieved_docs else "NO"
        answer = run.outputs.get("answer", "")
        prompt = (
            f"Relevant docs retrieved: {retrieval_status}\n"
            f"Response to evaluate: {answer}\n\n"
            f"{TECHNICAL_ACCURACY_JUDGE_PROMPT}\n"
            f"{self.parser.get_format_instructions()}"
        )
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResult(
            key="technical_accuracy",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class RetrievalQualityEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=RetrievalQualityEvaluation)
    def evaluate_run(self, run, example, **kwargs):
        issue_text = example.inputs.get("issue_text", "")
        answer = run.outputs.get("answer", "")
        
        # Check if the answer indicates no docs were retrieved
        if "I don't have enough information" in answer or "No relevant documentation found" in answer:
            docs_text = "No documents were retrieved."
        else:
            # Extract source URLs from the answer to indicate retrieval
            docs_text = "Documents were retrieved and used in the answer. Sources referenced: " + answer
            
        prompt = RETRIEVAL_QUALITY_JUDGE_PROMPT.format(
            issue_text=issue_text,
            retrieved_docs=docs_text
        ) + "\n" + self.parser.get_format_instructions()
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        return EvaluationResults(
            results=[
                EvaluationResult(
                    key="retrieval_relevance",
                    score=evaluation.relevance_score,
                    comment=evaluation.explanation,
                    evaluation_type="llm_judge"
                ),
                EvaluationResult(
                    key="retrieval_coverage",
                    score=evaluation.coverage_score,
                    comment=evaluation.explanation,
                    evaluation_type="llm_judge"
                )
            ]
        )

# ===== Main Execution =====
if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables before running the script.")
        sys.exit(1)
    
    # Print environment variables for debugging
    print(f"LANGSMITH_API_KEY exists: {bool(os.getenv('LANGSMITH_API_KEY'))}")
    print(f"LANGSMITH_PROJECT exists: {bool(os.getenv('LANGSMITH_PROJECT'))}")
    
    client = Client()

    def target(inputs: dict) -> dict:
        return triage_and_answer(
            issue_text=inputs["issue_text"],
            issue_url=inputs["issue_url"]
        )

    # Define all evaluators
    evaluators = [
        # Retrieval evaluator
        RetrievalQualityEvaluator(),
        # Classification evaluators
        IssueTypeEvaluator(),
        SeverityEvaluator(),
        # Response quality evaluators
        ResponseActionEvaluator(),
        ToneAppropriatenessEvaluator(),
        ResponseCompletenessEvaluator(),
        TechnicalAccuracyEvaluator()
    ]

    # Run the evaluation
    experiment_results = client.evaluate(
        target,
        data="CE Triage App: E2E",
        evaluators=evaluators,
        experiment_prefix="sdk:",
        max_concurrency=2,
    )

    print("Experiment results:", experiment_results)
    print("Check the LangSmith UI for the new experiment link!")
    
    # Run experiments by changing parameters
    # run_evaluation(
    #     experiment_name="improved-prompt",
    #     model_version="gpt-4"
    # ) 