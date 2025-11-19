# Requirements Files Explanation

This project provides multiple requirements files to accommodate different deployment scenarios and resource constraints.

## Files

### `requirements.txt` (Default - Lightweight)
The main requirements file contains only **essential dependencies** for running the API server with cloud LLM providers:

- **FastAPI & Core**: Web framework, config management, utilities
- **Cloud LLM APIs**: OpenAI, Anthropic, Google Gemini (lightweight, API-based)
- **LangChain**: Framework for LLM orchestration
- **Storage**: PostgreSQL, pgvector (ChromaDB optional)
- **Validation**: Basic NLP tools, safety checkers
- **Utilities**: Logging, async file handling

**Use when**: You want a fast Docker build and plan to use cloud APIs (OpenAI, Anthropic, etc.)

**Docker build time**: ~2-3 minutes
**Image size**: ~1.5-2 GB

### `requirements-full.txt` (Complete Features)
Includes everything from `requirements.txt` plus:

- **PyTorch & Transformers**: For local model inference
- **Spacy**: Advanced NLP processing
- **Datasets**: Training data loading
- **ChromaDB**: Vector database
- **Groq LangChain**: Additional provider support

**Use when**: You need local model support or advanced NLP features

**Docker build time**: ~10-15 minutes
**Image size**: ~5-6 GB

### `requirements-local.txt` (Local Inference Engines)
Adds C++-compiled inference engines:

- **llama-cpp-python**: GGUF model support
- **ctransformers**: GGML model support

**Note**: These require compilation and may fail on some systems without proper build tools.

**Use when**: You need llama.cpp or ctransformers specifically

## Installation Examples

### Minimal (Cloud APIs Only)
```bash
pip install -r requirements.txt
```

### Full Features (Local Models)
```bash
pip install -r requirements-full.txt
```

### With Local Inference Engines
```bash
pip install -r requirements.txt
pip install -r requirements-local.txt
```

## Docker Builds

### Lightweight (Default)
```bash
docker build -t llm-validator:lite --target application .
```

### Full Features
```bash
docker build -t llm-validator:full --target full-application .
```

### Docker Compose (Lightweight)
```bash
docker-compose up -d
```

## Troubleshooting

### Docker Build Fails
If you get `exit code: 1` during pip install:

1. **Use lightweight build** (default): Only installs essential packages
2. **Check network**: Large packages like PyTorch can timeout
3. **Increase resources**: Docker needs sufficient memory (>4GB recommended for full build)
4. **Use pre-built wheels**: Compilation packages may fail without build-essential

### Import Errors
If you get errors like "No module named 'torch'":

- **For cloud APIs**: You don't need torch, use the cloud providers
- **For local models**: Install `requirements-full.txt`
- **Check optional imports**: The code gracefully handles missing optional dependencies

## Recommended Deployments

### Production (Cloud APIs)
```dockerfile
# Use lightweight requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### Development (Full Features)
```dockerfile
# Use full requirements
COPY requirements-full.txt requirements.txt .
RUN pip install -r requirements-full.txt
```

### Research (Everything)
```bash
# Install all features locally
pip install -r requirements-full.txt
pip install -r requirements-local.txt
```

## Package Size Reference

| Category | Package | Size | Build Time |
|----------|---------|------|------------|
| Core | FastAPI, Pydantic | ~50MB | <1min |
| Cloud APIs | OpenAI, Anthropic | ~20MB | <1min |
| PyTorch | torch==2.1.2 | ~800MB | 2-5min |
| Transformers | transformers | ~400MB | 1-2min |
| llama.cpp | llama-cpp-python | ~50MB | 3-10min (compile) |
| ChromaDB | chromadb | ~200MB | 1-2min |

## Support

For build issues or questions, see:
- QUICKSTART.md - Quick setup guide
- README.md - Full documentation
- Dockerfile - Build configurations
