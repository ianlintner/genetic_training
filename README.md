# 🤖 Multi-LLM Validator with Genetic Optimization

A production-grade, modular LLM prompt/response validation system with multiple backends and pipelines, using Python, FastAPI, LangChain, local open-source models, and genetic optimization for scoring weights.

## ✨ Features

### 🎯 Core Capabilities

- **Multi-LLM Support**: OpenAI, Anthropic, Groq, Google Gemini, local Transformers, llama.cpp, ctransformers
- **Hybrid Validation**: Combines heuristic scoring with LLM-based judging
- **Adaptive Learning**: Learns from validation history to optimize thresholds and weights
- **Multiple Pipelines**: Generate+Validate, Validate-Only, and Revise Loop
- **Genetic Optimization**: Evolve optimal validator configurations using genetic algorithms
- **Flexible Storage**: JSON, SQLite, ChromaDB, or PostgreSQL+pgvector backends
- **REST API**: FastAPI with auto-generated OpenAPI documentation
- **Web Dashboard**: Simple HTML/JS interface for validation and training

### 🔍 Validation Heuristics

- Token length anomaly detection
- Contradiction detection
- Harmful content flagging
- Repetition analysis
- Structure compliance
- Hallucination pattern recognition

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- (Optional) Docker and Docker Compose
- API keys for cloud providers (OpenAI, Anthropic, etc.)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ianlintner/genetic_training.git
cd genetic_training
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.template .env
# Edit .env and add your API keys
```

5. **Run the API server**
```bash
python main.py api
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **With PostgreSQL**
```bash
docker-compose --profile with-postgres up -d
```

3. **With ChromaDB**
```bash
docker-compose --profile with-chromadb up -d
```

## 📖 Usage

### Command Line Interface

```bash
# Run API server
python main.py api

# Run training pipeline
python main.py train

# Run genetic evolution
python main.py evolve

# Validate text
python main.py validate "Your text here" --prompt "Original prompt"
```

### API Endpoints

#### Validation

```bash
# Validate existing text
curl -X POST http://localhost:8000/validate/single \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "prompt": "Original prompt"}'

# Generate and validate
curl -X POST http://localhost:8000/validate/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a summary of AI", "model": "gpt-4o-mini"}'

# Revision loop
curl -X POST http://localhost:8000/validate/revise \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "quality_threshold": 0.8}'
```

#### Training & Evolution

```bash
# Start training
curl -X POST http://localhost:8000/train/run \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "background": true}'

# Start evolution
curl -X POST http://localhost:8000/evolve/run \
  -H "Content-Type: application/json" \
  -d '{"population_size": 20, "generations": 50, "background": true}'
```

### Python API

```python
from app.validators import HybridValidator
from app.pipelines import GenerateValidatePipeline
from app.models import get_registry

# Validate text
validator = HybridValidator()
result = validator.validate("Your text here", prompt="Original prompt")
print(f"Score: {result.score}, Passed: {result.passed}")

# Generate and validate
pipeline = GenerateValidatePipeline(model_name="gpt-4o-mini")
result = pipeline.run("Write a summary of AI")
print(f"Generated: {result.text}")
print(f"Score: {result.validation.score}")

# Use specific model
registry = get_registry()
model = registry.get_model("claude-3-haiku-20240307")
text = model.generate("Hello, world!")
```

## 🎮 Web Dashboard

Access the web dashboard at `http://localhost:8000/frontend/index.html`

Features:
- Text validation interface
- Generate and validate forms
- Revision loop controls
- Training triggers
- Evolution configuration
- Score visualization

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

- Model selection and parameters
- Validation scoring weights
- Heuristic thresholds
- Pipeline settings
- Training configuration
- Evolution parameters
- Storage backend

Example configuration:

```yaml
models:
  selected_model: "gpt-4o-mini"
  
validation:
  scoring:
    weights:
      heuristics: 0.4
      judge: 0.6
    heuristics:
      harmful_content:
        enabled: true
        weight: 0.25
        threshold: 0.5

evolution:
  population_size: 20
  generations: 50
  mutation_rate: 0.2
```

## 📊 Architecture

```
┌─────────────────┐
│   FastAPI API   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│Models │ │Validators│
└───┬───┘ └──┬──────┘
    │        │
┌───▼────────▼───┐
│   Pipelines    │
└───┬────────┬───┘
    │        │
┌───▼───┐ ┌──▼───────┐
│Storage│ │ Training │
└───────┘ └──┬───────┘
             │
        ┌────▼────┐
        │Evolution│
        └─────────┘
```

### Components

- **Models**: Registry supporting multiple LLM backends
- **Validators**: Heuristics + LLM judge with adaptive learning
- **Pipelines**: Generation, validation, and revision workflows
- **Storage**: Pluggable backends for results persistence
- **Training**: Weight optimization from labeled datasets
- **Evolution**: Genetic algorithm for configuration optimization

## 🧬 Genetic Optimization

The system uses DEAP (Distributed Evolutionary Algorithms in Python) to evolve optimal validator configurations:

- **Chromosome**: Encodes weights, thresholds, and parameters
- **Mutation**: Randomly adjusts configuration values
- **Crossover**: Combines configurations from parent individuals
- **Fitness**: Evaluated on validation datasets (F1 score or error rate)
- **Selection**: Tournament selection with elitism

## 📦 Storage Backends

### JSON
Simple file-based storage for small datasets.

### SQLite
Embedded database for structured queries.

### ChromaDB
Vector database for semantic search and similarity.

### PostgreSQL + pgvector
Production-grade database with vector support.

## 🛠️ Development

### Project Structure

```
genetic_training/
├── app/
│   ├── models/          # LLM model implementations
│   ├── validators/      # Heuristics and judge
│   ├── pipelines/       # Generation and validation pipelines
│   ├── storage/         # Storage backends
│   ├── training/        # Training pipeline
│   ├── evolution/       # Genetic algorithm
│   ├── frontend/        # Web dashboard
│   ├── api.py          # FastAPI application
│   ├── base.py         # Base interfaces
│   └── config.py       # Configuration management
├── configs/
│   └── config.yaml     # Main configuration
├── tests/              # Unit tests
├── data/               # Data storage
├── logs/               # Application logs
├── models/             # Downloaded model files
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose setup
└── main.py           # CLI entry point
```

### Adding Custom Heuristics

```python
from app.base import BaseHeuristic

class MyHeuristic(BaseHeuristic):
    def validate(self, text: str, context=None) -> float:
        # Your validation logic
        score = 1.0  # 0.0 to 1.0
        return score
    
    def get_name(self) -> str:
        return "my_heuristic"
```

### Adding Model Backends

```python
from app.base import BaseModel, ModelBackend

class MyModel(BaseModel):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your generation logic
        return "Generated text"
    
    def get_model_name(self) -> str:
        return "my-model"
    
    def get_backend(self) -> ModelBackend:
        return ModelBackend.LOCAL_TRANSFORMERS
```

## 📝 API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with FastAPI, LangChain, and DEAP
- Inspired by research in LLM validation and quality assessment
- Uses multiple model providers for flexibility and comparison

## 📧 Contact

For questions or support, please open an issue on GitHub.
