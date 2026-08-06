# Start from a small official Python image
FROM python:3.12-slim

# Grab the uv binary from its official image (fast dependency installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy ONLY the dependency manifests first — Docker caches this layer,
# so deps reinstall only when they actually change (not on every code edit)
COPY pyproject.toml uv.lock ./

# Install runtime deps into /app/.venv (skip dev tools; don't install our package yet)
RUN uv sync --frozen --no-dev --no-install-project

# Now copy the rest (source, api/, models/)
COPY . .

# Install our own package into the venv
RUN uv sync --frozen --no-dev

# Make the venv the default Python, and let Python import the top-level `api/` package
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Document the port the app serves on
EXPOSE 8000

# Pull the trained model from the GitHub Release (it's DVC-tracked, not in git)
ADD https://github.com/Hamody22-gif/fraud-detection-mlops/releases/download/v0.1.0/xgboost.joblib /app/models/xgboost.joblib


# Start the server, bound to 0.0.0.0 so it's reachable from OUTSIDE the container
# (localhost inside a container is unreachable from your host — this is the #1 Docker gotcha)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
