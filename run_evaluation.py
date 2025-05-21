from langsmith import Client
from chains import triage_and_answer
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import os
import sys
from langsmith.evaluation import EvaluationResult, RunEvaluator

#read local .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# Evaluation Models
class ToneEvaluation(BaseModel):
    score: float = Field(description="Overall score from 0-1")
    explanation: str = Field(description="Step-by-step explanation of the score")
    professionalism: float = Field(description="Score for professionalism")
    empathy: float = Field(description="Score for empathy")
    clarity: float = Field(description="Score for clarity")
    positivity: float = Field(description="Score for positivity")

class CompletenessEvaluation(BaseModel):
    score: float = Field(description="Overall score from 0-1")
    explanation: str = Field(description="Step-by-step explanation of the score")
    technical_details: float = Field(description="Score for technical details")
    explanation_quality: float = Field(description="Score for explanation quality")
    next_steps: float = Field(description="Score for next steps")
    references: float = Field(description="Score for references and links")

class TechnicalAccuracyEvaluation(BaseModel):
    score: float = Field(description="Overall score from 0-1")
    explanation: str = Field(description="Step-by-step explanation of the score")
    code_accuracy: float = Field(description="Score for code references accuracy")
    doc_accuracy: float = Field(description="Score for documentation accuracy")
    technical_terms: float = Field(description="Score for technical terminology")

# Evaluation Prompts
TONE_EVALUATION_PROMPT = """You are evaluating the tone of a customer support response.

Consider:
1. Professionalism - Is the response professional and appropriate?
2. Empathy - Does it show understanding of the user's issue?
3. Clarity - Is the response clear and easy to understand?
4. Positivity - Does it maintain a positive, helpful tone?

Score from 0-1 where:
1.0 = Perfect professional tone
0.0 = Unprofessional or inappropriate tone

Explain your reasoning step by step."""

COMPLETENESS_EVALUATION_PROMPT = """You are evaluating the completeness of a customer support response.

Consider:
1. Technical Details - Does it include necessary technical information?
2. Explanation Quality - Is the explanation thorough and clear?
3. Next Steps - Does it provide clear next steps or solutions?
4. References - Does it include relevant documentation links?

Score from 0-1 where:
1.0 = Complete response with all necessary information
0.0 = Incomplete or missing critical information

Explain your reasoning step by step."""

TECHNICAL_ACCURACY_PROMPT = """You are evaluating the technical accuracy of a customer support response.

Consider:
1. Code References - Are code examples or references accurate?
2. Documentation - Are documentation links and references correct?
3. Technical Terms - Is technical terminology used correctly?

Score from 0-1 where:
1.0 = Technically accurate in all aspects
0.0 = Contains technical inaccuracies

Explain your reasoning step by step."""

# Primary Evaluators (Accuracy-based)
class IssueTypeEvaluator(RunEvaluator):
    """Evaluates the accuracy of issue type classification."""
    def evaluate_run(self, run, example, **kwargs):
        predicted = run.outputs.get("issue_type", "")
        expected = example.outputs.get("issue_type", "")
        score = 1.0 if predicted.lower() == expected.lower() else 0.0
        return EvaluationResult(
            key="issue_type_accuracy",
            score=score,
            comment=f"Predicted: {predicted}, Expected: {expected}",
            evaluation_type="accuracy"
        )

class SeverityEvaluator(RunEvaluator):
    """Evaluates the accuracy of severity score prediction."""
    def evaluate_run(self, run, example, **kwargs):
        predicted = run.outputs.get("severity", "")
        expected = example.outputs.get("severity", 0)
        
        # Extract severity number from formatted string
        try:
            if "severity" in str(predicted).lower():
                predicted_num = int(str(predicted).split("severity")[1].strip().split()[0])
            else:
                predicted_num = int(str(predicted).strip().split()[0])
        except (ValueError, IndexError):
            predicted_num = 0
            
        score = 1.0 if predicted_num == expected else 0.0
        return EvaluationResult(
            key="severity_accuracy",
            score=score,
            comment=f"Predicted: {predicted_num}, Expected: {expected}",
            evaluation_type="accuracy"
        )

class ResponseActionEvaluator(RunEvaluator):
    """Evaluates if the response correctly addresses the issue."""
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        issue_type = run.outputs.get("issue_type", "")
        severity = run.outputs.get("severity", "")
        
        # Extract severity number from formatted string
        try:
            if "severity" in str(severity).lower():
                severity_num = int(str(severity).split("severity")[1].strip().split()[0])
            else:
                severity_num = int(str(severity).strip().split()[0])
        except (ValueError, IndexError):
            severity_num = 0
        
        # Check if response matches the severity and type
        has_appropriate_action = (
            (severity_num >= 3 and "urgent" in answer.lower()) or
            (severity_num <= 2 and "can be addressed" in answer.lower())
        )
        
        has_type_specific_guidance = (
            (issue_type == "bug report" and "fix" in answer.lower()) or
            (issue_type == "feature request" and "implement" in answer.lower()) or
            (issue_type == "question" and "answer" in answer.lower())
        )
        
        score = sum([has_appropriate_action, has_type_specific_guidance]) / 2.0
        return EvaluationResult(
            key="response_action_accuracy",
            score=score,
            comment=f"Appropriate action: {has_appropriate_action}, Type-specific guidance: {has_type_specific_guidance}",
            evaluation_type="accuracy"
        )

# Supplementary Evaluators (LLM-as-judge)
class ToneAppropriatenessEvaluator(RunEvaluator):
    """Uses LLM to evaluate if the response maintains appropriate professional tone."""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=ToneEvaluation)
    
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        prompt = f"""Response to evaluate: {answer}\n\n{TONE_EVALUATION_PROMPT}\n{self.parser.get_format_instructions()}"""
        
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        
        return EvaluationResult(
            key="tone_appropriateness",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class ResponseCompletenessEvaluator(RunEvaluator):
    """Uses LLM to evaluate if the response is complete and comprehensive."""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=CompletenessEvaluation)
    
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        prompt = f"""Response to evaluate: {answer}\n\n{COMPLETENESS_EVALUATION_PROMPT}\n{self.parser.get_format_instructions()}"""
        
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        
        return EvaluationResult(
            key="response_completeness",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

class TechnicalAccuracyEvaluator(RunEvaluator):
    """Uses LLM to evaluate the technical accuracy of the response."""
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=TechnicalAccuracyEvaluation)
    
    def evaluate_run(self, run, example, **kwargs):
        answer = run.outputs.get("answer", "")
        prompt = f"""Response to evaluate: {answer}\n\n{TECHNICAL_ACCURACY_PROMPT}\n{self.parser.get_format_instructions()}"""
        
        result = self.llm.invoke(prompt)
        evaluation = self.parser.parse(result.content)
        
        return EvaluationResult(
            key="technical_accuracy",
            score=evaluation.score,
            comment=evaluation.explanation,
            evaluation_type="llm_judge"
        )

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

    evaluators = [
        IssueTypeEvaluator(),
        SeverityEvaluator(),
        ResponseActionEvaluator(),
        ToneAppropriatenessEvaluator(),
        ResponseCompletenessEvaluator(),
        TechnicalAccuracyEvaluator()
    ]

    experiment_results = client.evaluate(
        target,
        data="CE Triage App: E2E",
        evaluators=evaluators,
        experiment_prefix="baseline",
        max_concurrency=2,
    )

    print("Experiment results:", experiment_results)
    print("Check the LangSmith UI for the new experiment link!")
    
    # You can run additional experiments by changing parameters
    # run_evaluation(
    #     experiment_name="improved-prompt",
    #     model_version="gpt-4"
    # ) 