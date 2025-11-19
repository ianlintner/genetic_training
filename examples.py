"""
Example usage of the Multi-LLM Validator system.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_config
from app.models import get_registry
from app.validators import HybridValidator
from app.pipelines import GenerateValidatePipeline, ValidateOnlyPipeline, ReviseLoopPipeline
from loguru import logger


def example_validate_text():
    """Example: Validate existing text."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Validate Existing Text")
    print("="*60)
    
    validator = HybridValidator()
    
    text = """
    Artificial Intelligence is a revolutionary technology that enables machines 
    to perform tasks that typically require human intelligence. It has 
    applications in healthcare, finance, education, and many other fields.
    """
    
    result = validator.validate(text, prompt="Explain AI")
    
    print(f"\nValidation Score: {result.score:.3f}")
    print(f"Passed: {result.passed}")
    print(f"\nHeuristic Scores:")
    for name, score in result.heuristic_scores.items():
        print(f"  {name}: {score:.3f}")
    print(f"\nJudge Score: {result.judge_score:.3f}")


def example_generate_validate():
    """Example: Generate and validate text."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate and Validate")
    print("="*60)
    
    # Note: This requires API keys to be configured
    try:
        pipeline = GenerateValidatePipeline(model_name="gpt-4o-mini")
        
        prompt = "Write a brief explanation of machine learning."
        print(f"\nPrompt: {prompt}")
        
        result = pipeline.run(prompt)
        
        print(f"\nGenerated Text:\n{result.text}")
        print(f"\nValidation Score: {result.validation.score:.3f}")
        print(f"Passed: {result.validation.passed}")
        
    except Exception as e:
        print(f"\nNote: Generation requires API keys. Error: {e}")


def example_revise_loop():
    """Example: Use revision loop to improve quality."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Revision Loop")
    print("="*60)
    
    try:
        pipeline = ReviseLoopPipeline(
            model_name="gpt-4o-mini",
            max_revisions=3,
            quality_threshold=0.8
        )
        
        prompt = "Explain the concept of neural networks."
        print(f"\nPrompt: {prompt}")
        print("Running revision loop...")
        
        result = pipeline.run(prompt)
        
        print(f"\nFinal Text:\n{result.text}")
        print(f"\nFinal Score: {result.validation.score:.3f}")
        print(f"\nRevision History:")
        for rev in result.metadata.get("revision_history", []):
            print(f"  Revision {rev['revision']}: {rev['score']:.3f} ({'✓' if rev['passed'] else '✗'})")
        
    except Exception as e:
        print(f"\nNote: Revision requires API keys. Error: {e}")


def example_custom_heuristic():
    """Example: Using custom validation configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    from app.validators.heuristics import TokenLengthHeuristic, RepetitionHeuristic
    
    # Create custom validator
    validator = HybridValidator()
    
    # Access and configure heuristics
    print("\nHeuristics in use:")
    for heuristic in validator.heuristics:
        print(f"  - {heuristic.get_name()}: weight={heuristic.weight}")
    
    # Validate with custom settings
    text = "This is a test. This is a test. This is a test."
    result = validator.validate(text)
    
    print(f"\nValidation of repetitive text:")
    print(f"Overall Score: {result.score:.3f}")
    print(f"Repetition Score: {result.heuristic_scores.get('repetition', 'N/A')}")


def example_storage():
    """Example: Using storage backends."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Storage Backends")
    print("="*60)
    
    from app.storage import create_store
    
    # Create JSON store
    store = create_store("json", {"path": "data/example_results.json"})
    
    # Save a record
    record = {
        "text": "Sample text",
        "score": 0.85,
        "passed": True,
        "type": "example"
    }
    
    record_id = store.save(record)
    print(f"\nSaved record with ID: {record_id}")
    
    # Query records
    records = store.query(filters={"type": "example"}, limit=10)
    print(f"Found {len(records)} example records")
    
    # Get by ID
    retrieved = store.get_by_id(record_id)
    print(f"Retrieved record: {retrieved.get('text')}")
    
    store.close()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Multi-LLM Validator - Examples")
    print("="*60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    print(f"Selected model: {config.models.get('selected_model')}")
    
    # Run examples
    example_validate_text()
    example_custom_heuristic()
    example_storage()
    
    # Examples that require API keys
    print("\n\nNote: The following examples require API keys to be configured in .env file:")
    print("  - Generate and Validate")
    print("  - Revision Loop")
    print("\nUncomment the lines below to run them:")
    
    # example_generate_validate()
    # example_revise_loop()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
