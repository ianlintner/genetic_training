# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ianlintner/genetic_training.git
cd genetic_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your API keys (optional for testing)
nano .env
```

### 3. Run Examples (No API Keys Required)

```bash
# Run basic validation examples
python examples.py
```

**Expected Output:**
```
Testing imports...
✓ Config module loaded
✓ Heuristics loaded
✓ Base classes loaded
✓ Token length heuristic: score=1.000
✓ Repetition heuristic: score=1.000

✅ All basic tests passed!
```

### 4. Start API Server

```bash
# Start the FastAPI server
python main.py api
```

Then open your browser to:
- **Dashboard**: http://localhost:8000/frontend/index.html
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 5. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/

# Validate text (no API key needed for heuristics)
curl -X POST http://localhost:8000/validate/single \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a test message for validation."}'
```

### 6. Using Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## 📖 Key Commands

### CLI Commands

```bash
# Run API server
python main.py api

# Run training (requires datasets or uses mock data)
python main.py train

# Run genetic evolution
python main.py evolve

# Validate text from command line
python main.py validate "Your text here" --prompt "Original prompt"
```

### Docker Commands

```bash
# Basic deployment
docker-compose up -d

# With PostgreSQL
docker-compose --profile with-postgres up -d

# With ChromaDB
docker-compose --profile with-chromadb up -d

# View logs
docker-compose logs -f

# Restart service
docker-compose restart api

# Stop all services
docker-compose down
```

## 🧪 Testing Features

### 1. Heuristic Validation (No API Keys)

```python
from app.validators.heuristics import TokenLengthHeuristic

heuristic = TokenLengthHeuristic(min_tokens=5, max_tokens=100)
score = heuristic.validate("Your text here")
print(f"Score: {score}")
```

### 2. Storage

```python
from app.storage import JSONStore

store = JSONStore("data/my_results.json")
record_id = store.save({"text": "Test", "score": 0.85})
record = store.get_by_id(record_id)
```

### 3. With API Keys

```python
from app.models import get_registry
from app.pipelines import GenerateValidatePipeline

# Generate and validate
pipeline = GenerateValidatePipeline(model_name="gpt-4o-mini")
result = pipeline.run("Explain quantum computing")
print(f"Generated: {result.text}")
print(f"Score: {result.validation.score}")
```

## 🎯 Common Use Cases

### Validate User-Generated Content

```bash
curl -X POST http://localhost:8000/validate/single \
  -H "Content-Type: application/json" \
  -d '{"text": "User comment here", "prompt": "Write a review"}'
```

### Generate High-Quality Content

```bash
curl -X POST http://localhost:8000/validate/revise \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a product description", 
    "quality_threshold": 0.85,
    "max_revisions": 3
  }'
```

### Train Custom Weights

```bash
curl -X POST http://localhost:8000/train/run \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "background": true}'
```

### Optimize Configuration with Evolution

```bash
curl -X POST http://localhost:8000/evolve/run \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 20, 
    "generations": 50,
    "background": true
  }'
```

## 🔧 Configuration

### Model Selection

Edit `configs/config.yaml`:

```yaml
models:
  selected_model: "gpt-4o-mini"  # Change to any supported model
```

### Validation Weights

```yaml
validation:
  scoring:
    weights:
      heuristics: 0.4  # 40% heuristic scoring
      judge: 0.6       # 60% LLM judge scoring
```

### Heuristic Thresholds

```yaml
validation:
  scoring:
    heuristics:
      harmful_content:
        enabled: true
        weight: 0.25
        threshold: 0.5
```

## 📊 Monitoring

### Check Stats

```bash
curl http://localhost:8000/stats
```

### View Configuration

```bash
curl http://localhost:8000/config
```

### List Available Models

```bash
curl http://localhost:8000/models/list
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### API Key Issues

```bash
# Check .env file exists and has keys
cat .env

# Keys should be in format:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Port Already in Use

```bash
# Change port in configs/config.yaml
api:
  port: 8001  # Use different port
```

### Docker Issues

```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📚 Next Steps

1. **Read the full README.md** for detailed documentation
2. **Explore the examples.py** for usage patterns
3. **Check the API docs** at http://localhost:8000/docs
4. **Customize configs/config.yaml** for your use case
5. **Add custom heuristics** by extending BaseHeuristic
6. **Integrate with your application** using the Python API

## 🆘 Getting Help

- **Documentation**: See README.md
- **Examples**: Run `python examples.py`
- **API Reference**: http://localhost:8000/docs
- **Issues**: Open an issue on GitHub

## ✨ Key Features to Try

1. ✅ **Validate text** without any API keys using heuristics
2. 🤖 **Generate & validate** with multiple LLM providers
3. 🔄 **Revision loops** for iterative improvement
4. 🎓 **Train weights** on your own datasets
5. 🧬 **Evolve configurations** with genetic algorithms
6. 💾 **Store results** in JSON, SQLite, or ChromaDB
7. 🌐 **Use the web dashboard** for visual interaction
8. 🐳 **Deploy with Docker** in one command
