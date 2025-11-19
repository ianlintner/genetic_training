"""
Prompt templates for LLM validation and revision.
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of AI-generated text.

Evaluate the following response based on these criteria:
1. Correctness: Is the information accurate and appropriate?
2. Coherence: Is the text well-structured and logical?
3. Hallucination: Does it contain false or fabricated information?
4. Completeness: Does it fully address the prompt?

Prompt: {prompt}

Response: {response}

Provide your evaluation as a JSON object with scores from 0.0 to 1.0 for each criterion:
{{
  "correctness": 0.0-1.0,
  "coherence": 0.0-1.0,
  "hallucination_free": 0.0-1.0,
  "completeness": 0.0-1.0,
  "overall": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Only respond with the JSON object, no other text."""


REVISION_PROMPT_TEMPLATE = """The following response needs improvement:

Original Prompt: {original_prompt}

Current Response: {text}

Issues identified:
{issues}

Judge feedback: {reasoning}

Please provide an improved version that addresses these issues while maintaining the core content and intent."""
