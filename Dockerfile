FROM python:3.13-slim       
# Base image - slim = smaller size

# All subsequent commands run here
WORKDIR /app                

# Installing uv in the image
RUN pip install uv          

# Copy dependecy files
COPY pyproject.toml uv.lock README.md ./

# Dependencies installations
RUN uv pip install --system -r pyproject.toml

# Copying source code
COPY src/ ./src/

COPY data/vectorstore/ ./data/vectorstore/

# Install package itself
RUN uv pip install --system --no-deps .

# Installing the embedder model 
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "amf_rag_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]