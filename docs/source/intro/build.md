# Build

## Building and Installing Context Aware RAG

### Prerequisites

Before you begin using Context Aware RAG, ensure that you meet the following software prerequisites.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

1. Clone the repository

```bash
git clone git@github.com:NVIDIA/context-aware-rag.git
cd context-aware-rag/
```

2. Create a Python environment

```bash
uv venv --seed .venv
source .venv/bin/activate
```

3. (A) Install the Context Aware RAG library from source

```bash
uv pip install -e .
```

3. (B) Build and install from wheel file
```bash
uv build
uv pip install dist/vss-ctx-rag-1.0.2-py3-none-any.whl
```
