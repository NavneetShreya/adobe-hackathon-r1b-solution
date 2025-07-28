# # # # # Stage 1: Builder - Install dependencies and download SentenceTransformer model
# # # # FROM --platform=linux/amd64 python:3.9-slim-bookworm AS builder

# # # # WORKDIR /app

# # # # # Clean apt cache, update, and install system dependencies for PyMuPDF and others
# # # # RUN apt-get clean && \
# # # #     apt-get update && \
# # # #     apt-get install -y --no-install-recommends \
# # # #     build-essential \
# # # #     libjpeg-dev \
# # # #     zlib1g-dev \
# # # #     pkg-config \
# # # #     libpng-dev \
# # # #     && rm -rf /var/lib/apt/lists/*

# # # # COPY requirements.txt .
# # # # RUN pip install --no-cache-dir -r requirements.txt

# # # # # Download the Sentence Transformer model during the build process
# # # # # This ensures the model is available offline at runtime and is part of the image.
# # # # # The model will be cached in /root/.cache/torch/sentence_transformers by default.
# # # # ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers
# # # # RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# # # # # Stage 2: Final image - Copy only necessary runtime files
# # # # FROM --platform=linux/amd64 python:3.9-slim-bookworm

# # # # WORKDIR /app

# # # # # Clean apt cache, update, and install only runtime system dependencies
# # # # RUN apt-get clean && \
# # # #     apt-get update && \
# # # #     apt-get install -y --no-install-recommends \
# # # #     libjpeg-turbo-progs \
# # # #     zlib1g \
# # # #     libpng16-16 \
# # # #     && rm -rf /var/lib/apt/lists/*

# # # # COPY main.py .
# # # # # Copy the downloaded Sentence Transformer model from the builder stage's cache
# # # # COPY --from=builder /root/.cache/torch/sentence_transformers /root/.cache/torch/sentence_transformers

# # # # # Ensure the required Python packages are installed in the final image
# # # # COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# # # # COPY --from=builder /usr/local/bin /usr/local/bin

# # # # CMD ["python", "main.py"]


# # # ### 3. `adobe-hackathon-r1b-solution/Dockerfile`

# # # # This `Dockerfile` is robust for Round 1B, including the Sentence Transformer model download.


# # # # ```dockerfile
# # # # Stage 1: Builder - Install dependencies and download SentenceTransformer model
# # # FROM --platform=linux/amd64 python:3.9-slim-bookworm AS builder

# # # WORKDIR /app

# # # # Clean apt cache, update, and install system dependencies for PyMuPDF and others
# # # RUN apt-get clean && \
# # #     apt-get update && \
# # #     apt-get install -y --no-install-recommends \
# # #     build-essential \
# # #     libjpeg-dev \
# # #     zlib1g-dev \
# # #     pkg-config \
# # #     libpng-dev \
# # #     && rm -rf /var/lib/apt/lists/*

# # # COPY requirements.txt .
# # # RUN pip install --no-cache-dir -r requirements.txt

# # # # Download the Sentence Transformer model during the build process
# # # # This ensures the model is available offline at runtime and is part of the image.
# # # # The model will be cached in /root/.cache/torch/sentence_transformers by default.
# # # ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers
# # # RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# # # # Stage 2: Final image - Copy only necessary runtime files
# # # FROM --platform=linux/amd64 python:3.9-slim-bookworm

# # # WORKDIR /app

# # # # Clean apt cache, update, and install only runtime system dependencies
# # # RUN apt-get clean && \
# # #     apt-get update && \
# # #     apt-get install -y --no-install-recommends \
# # #     libjpeg-turbo-progs \
# # #     zlib1g \
# # #     libpng16-16 \
# # #     && rm -rf /var/lib/apt/lists/*

# # # COPY main.py .
# # # # Copy the downloaded Sentence Transformer model from the builder stage's cache
# # # COPY --from=builder /root/.cache/torch/sentence_transformers /root/.cache/torch/sentence_transformers

# # # # Ensure the required Python packages are installed in the final image
# # # COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# # # COPY --from=builder /usr/local/bin /usr/local/bin

# # # CMD ["python", "main.py"]


# # # Stage 1: Builder - Install build tools and download models/wheels
# # FROM --platform=linux/amd64 python:3.9-slim-bookworm AS builder

# # WORKDIR /app

# # # Set environment variables for strict offline mode for Hugging Face Hub operations
# # ENV HF_HUB_OFFLINE=1
# # ENV TRANSFORMERS_OFFLINE=1

# # # Install system build dependencies required for PyMuPDF and others
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     build-essential \
# #     libjpeg-dev \
# #     zlib1g-dev \
# #     pkg-config \
# #     libpng-dev \
# #     git \
# #     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # # Install Python dependencies into a specific temporary directory
# # ENV PYTHON_DEPENDENCIES_PATH=/tmp/python_dependencies
# # RUN mkdir -p ${PYTHON_DEPENDENCIES_PATH}

# # # Install PyTorch CPU-only explicitly into the target directory
# # # This installation bypasses default PyPI resolution which can be slow/large.
# # RUN pip install --no-cache-dir --target=${PYTHON_DEPENDENCIES_PATH} \
# #     torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# # # Install other requirements into the same target directory
# # COPY requirements.txt .
# # RUN pip install --no-cache-dir --target=${PYTHON_DEPENDENCIES_PATH} -r requirements.txt

# # # Download NLTK data to a specific cache location
# # ENV NLTK_DATA_PATH=/tmp/nltk_data
# # RUN python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA_PATH}')"

# # # Download Sentence Transformer model to a specific cache location, forcing local_files_only=True via env vars
# # ENV SENTENCE_TRANSFORMERS_MODEL_PATH=/tmp/sentence_transformers_model
# # RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='${SENTENCE_TRANSFORMERS_MODEL_PATH}')"

# # # Stage 2: Final image - Minimal runtime environment
# # FROM --platform=linux/amd64 python:3.9-slim-bookworm

# # WORKDIR /app

# # # Set environment variables for strict offline mode in the final runtime environment
# # ENV HF_HUB_OFFLINE=1
# # ENV TRANSFORMERS_OFFLINE=1

# # # Install only runtime system libraries needed by the Python packages
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     libjpeg-turbo-progs \
# #     zlib1g \
# #     libpng16-16 \
# #     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # # Copy the Python dependencies from the builder stage to site-packages
# # # This copies only the necessary installed files, reducing final image size significantly.
# # COPY --from=builder ${PYTHON_DEPENDENCIES_PATH} /usr/local/lib/python3.9/site-packages/

# # # Copy NLTK data
# # COPY --from=builder ${NLTK_DATA_PATH} /usr/local/nltk_data/
# # ENV NLTK_DATA=/usr/local/nltk_data

# # # Copy Sentence Transformer model
# # COPY --from=builder ${SENTENCE_TRANSFORMERS_MODEL_PATH} /root/.cache/torch/sentence_transformers/
# # ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

# # # Copy application code
# # COPY main.py .

# # # Set PYTHONPATH to include the copied packages (usually not strictly needed if in site-packages, but good for safety)
# # ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:${PYTHONPATH}

# # CMD ["python", "main.py"]


# # Stage 1: Builder - Install build tools and download models/wheels
# FROM --platform=linux/amd64 python:3.9-slim-bookworm AS builder

# WORKDIR /app

# # Set environment variables for strict offline mode for Hugging Face Hub operations
# ENV HF_HUB_OFFLINE=1
# ENV TRANSFORMERS_OFFLINE=1

# # Install system build dependencies required for PyMuPDF and others
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     libjpeg-dev \
#     zlib1g-dev \
#     pkg-config \
#     libpng-dev \
#     git \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies into a specific temporary directory
# ENV PYTHON_DEPENDENCIES_PATH=/tmp/python_dependencies
# RUN mkdir -p ${PYTHON_DEPENDENCIES_PATH}

# # Install PyTorch CPU-only explicitly into the target directory
# # This installation bypasses default PyPI resolution which can be slow/large.
# RUN pip install --no-cache-dir --target=${PYTHON_DEPENDENCIES_PATH} \
#     torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# # Install other requirements into the same target directory
# COPY requirements.txt .
# RUN pip install --no-cache-dir --target=${PYTHON_DEPENDENCIES_PATH} -r requirements.txt

# # --- FIX: NLTK download command moved AFTER pip install of requirements.txt ---
# # NLTK data download
# ENV NLTK_DATA_PATH=/tmp/nltk_data
# RUN PYTHONPATH=${PYTHON_DEPENDENCIES_PATH} python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA_PATH}')"
# # --- END FIX ---

# # Download Sentence Transformer model to a specific cache location, forcing local_files_only=True via env vars
# ENV SENTENCE_TRANSFORMERS_MODEL_PATH=/tmp/sentence_transformers_model
# RUN PYTHONPATH=${PYTHON_DEPENDENCIES_PATH} python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='${SENTENCE_TRANSFORMERS_MODEL_PATH}')"

# # Stage 2: Final image - Minimal runtime environment
# FROM --platform=linux/amd64 python:3.9-slim-bookworm

# WORKDIR /app

# # Set environment variables for strict offline mode in the final runtime environment
# ENV HF_HUB_OFFLINE=1
# ENV TRANSFORMERS_OFFLINE=1

# # Install only runtime system libraries needed by the Python packages
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libjpeg-turbo-progs \
#     zlib1g \
#     libpng16-16 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Copy the Python dependencies from the builder stage to site-packages
# # This copies only the necessary installed files, reducing final image size significantly.
# COPY --from=builder ${PYTHON_DEPENDENCIES_PATH} /usr/local/lib/python3.9/site-packages/

# # Copy NLTK data
# COPY --from=builder ${NLTK_DATA_PATH} /usr/local/nltk_data/
# ENV NLTK_DATA=/usr/local/nltk_data

# # Copy Sentence Transformer model
# COPY --from=builder ${SENTENCE_TRANSFORMERS_MODEL_PATH} /root/.cache/torch/sentence_transformers/
# ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

# # Copy application code
# COPY main.py .

# # Set PYTHONPATH to include the copied packages (usually not strictly needed if in site-packages, but good for safety)
# ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:${PYTHONPATH}

# CMD ["python", "main.py"]


# Stage 1: Builder - Only download models and NLTK data
FROM --platform=linux/amd64 python:3.9-slim-bookworm AS builder

WORKDIR /app

# Temporarily set offline flags to 0 during build to allow downloads.
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# Install system dependencies needed for git and basic tools (e.g., for model downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Temporarily install NLTK in the builder stage to enable nltk.download
RUN pip install --no-cache-dir nltk==3.8.1

# Download NLTK data to a temporary cache location
ENV NLTK_DATA_PATH=/tmp/nltk_data_cache
RUN python -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA_PATH}')"

# Temporarily install sentence-transformers in the builder stage to enable model download
RUN pip install --no-cache-dir sentence-transformers==2.7.0

# Download Sentence Transformer model to a specific cache location.
# This step requires internet access during Docker build.
ENV SENTENCE_TRANSFORMERS_MODEL_PATH=/tmp/sentence_transformers_model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='${SENTENCE_TRANSFORMERS_MODEL_PATH}')"


# Stage 2: Final image - Minimal runtime environment with direct package installation
FROM --platform=linux/amd64 python:3.9-slim-bookworm

WORKDIR /app

# Set environment variables for strict offline mode in the final runtime environment.
# These flags ensure no network calls are made ONCE THE IMAGE IS BUILT AND RUN.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Install runtime system libraries AND build dependencies for PyMuPDF directly in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-turbo-progs \
    zlib1g \
    libpng16-16 \
    build-essential \
    libjpeg-dev \
    pkg-config \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only explicitly first, using its specific index.
# This must be separate from the requirements.txt installation.
RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy requirements.txt. Ensure it DOES NOT contain 'torch'.
COPY requirements.txt .
# Install remaining packages from requirements.txt (they will use default PyPI)
RUN pip install --no-cache-dir -r requirements.txt

# Copy NLTK data from the builder stage
COPY --from=builder ${NLTK_DATA_PATH} /usr/local/nltk_data/
ENV NLTK_DATA=/usr/local/nltk_data

# Copy Sentence Transformer model from the builder stage
COPY --from=builder ${SENTENCE_TRANSFORMERS_MODEL_PATH} /root/.cache/torch/sentence_transformers/
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/torch/sentence_transformers

# Copy application code
COPY main.py .

# Set PYTHONPATH (usually not strictly needed if in /usr/local/lib/python3.9/site-packages, but good for safety)
ENV PYTHONPATH=/usr/local/lib/python3.9/site-packages:${PYTHONPATH}

CMD ["python", "main.py"]