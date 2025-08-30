# ./Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Build deps for statsmodels/pmdarima (LAPACK/BLAS), matplotlib fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran \
    libatlas-base-dev liblapack-dev libopenblas-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install your package 
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# If you add a console script entrypoint (see pyproject snippet below),
# this will run `forecast-models` by default (prints CLI help).
ENTRYPOINT ["forecast-models"]
CMD ["--help"]
