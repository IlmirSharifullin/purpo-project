FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no venv needed inside container)
RUN uv sync --frozen --no-dev --no-cache

# Copy project source
COPY . .

# Create data and results directories
RUN mkdir -p data evals_results

# Default: interactive REPL
CMD ["uv", "run", "python", "main.py"]
