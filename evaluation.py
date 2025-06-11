from pydantic import BaseModel
import google.generativeai as genai
import instructor
from deepeval.models import DeepEvalBaseLLM
from deepeval import evaluate
from deepeval.metrics import (
    ContextualRecallMetric, 
    AnswerRelevancyMetric, 
    FaithfulnessMetric, 
    ContextualPrecisionMetric, 
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase

# Simple function to calculate token estimates
def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in the given text."""
    # Assuming 1 token ~ 4 characters (standard approximation for English text)
    return len(text.split())  # Alternative: len(text) // 4

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        # Initialize the Gemini 1.5 Flash model
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    def load_model(self):
        # Load and return the Gemini model instance
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # Load the model
        client = self.load_model()

        # Initialize the instructor client to work with structured responses
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )

        # Make a request with the given prompt and response schema
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )

        # Return just the response, as deepeval expects this
        return resp

    def generate_with_tokens(self, prompt: str, schema: BaseModel) -> tuple[BaseModel, dict]:
        """
        This method returns both the response and token usage.
        It is only used when you explicitly want to track tokens.
        """
        # Load the model
        client = self.load_model()

        # Initialize the instructor client
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )

        # Estimate input tokens
        input_tokens = estimate_tokens(prompt)

        # Generate response
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )

        # Estimate output tokens
        output_tokens = estimate_tokens(str(resp))

        return resp, {"input_tokens": input_tokens, "output_tokens": output_tokens}

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"

# Configure Gemini API
genai.configure(api_key='AIzaSyAVQ6HVewM3Uv1e5zguKwpZljAjV2rI_jU')

# Initialize the custom Gemini model
custom_gemini = CustomGeminiFlash()

# Replace this with the actual output from your LLM application
actual_output = "All items purchased online can be returned within 30 days for a full refund."
expected_output = "Online purchases are eligible for a full refund if returned within 30 days."
retrieval_context = ["Our return policy states that items bought online are eligible for a full refund within 30 days of purchase."]


# Define the schema for structured responses
class ResponseSchema(BaseModel):
    answer: str

# Generate output with token usage tracking
response, token_usage = custom_gemini.generate_with_tokens(
    prompt="What if these shoes don't fit?",
    schema=ResponseSchema
)

# Print the response and token usage
print("Generated Response:", response)
print("Token Usage:", token_usage)

# Create the test case
test_case = LLMTestCase(
    input="Can I return an online purchase?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# Define all metrics
metrics = [
    ContextualRecallMetric(
        threshold=0.7,
        model=custom_gemini,
        include_reason=True
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        model=custom_gemini,
        include_reason=True
    ),
    FaithfulnessMetric(
        threshold=0.7,
        model=custom_gemini,
        include_reason=True
    ),
    ContextualPrecisionMetric(
        threshold=0.7,
        model=custom_gemini,
        include_reason=True
    ),
    ContextualRelevancyMetric(
        threshold=0.7,
        model=custom_gemini,
        include_reason=True
    )
]

# Evaluate test cases in bulk
results = evaluate([test_case], metrics)
print(results,"evaluate result")

# Print results
for metric, result in zip(metrics, results):
    print(f"Metric: {metric.__class__.__name__}")
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

