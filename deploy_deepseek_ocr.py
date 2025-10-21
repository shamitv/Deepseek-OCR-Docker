import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# --- Configuration ---
SCRIPT_VERSION = "1.4"
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
VLLM_REPO = "https://github.com/vllm-project/vllm.git"
DOCKER_BASE_IMAGE = "ubuntu:22.04"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def run_command(command, cwd=None, capture_output=False, text=True):
    """
    Executes a shell command and logs its output, raising an error on failure.
    """
    if isinstance(command, str):
        command = command.split()

    logging.info(f"Running command: {' '.join(command)}" + (f" in {cwd}" if cwd else ""))
    try:
        result = subprocess.run(
            command,
            check=True,
            cwd=cwd,
            capture_output=capture_output,
            text=text
        )
        if capture_output:
            logging.info(f"Command output:\n{result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr}")
        raise


def download_model(model_weights_dir: Path):
    """
    Downloads the model from Hugging Face if it doesn't already exist.
    """
    if model_weights_dir.exists() and any(model_weights_dir.iterdir()):
        logging.warning(
            f"Destination directory '{model_weights_dir}' already exists and is not empty. Skipping download."
        )
        return

    logging.info(f"Downloading model '{MODEL_ID}' to '{model_weights_dir}'...")
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    snapshot_download(
        MODEL_ID,
        local_dir=str(model_weights_dir),
        local_dir_use_symlinks=False,
        tqdm_class=tqdm,
    )
    logging.info("Model download complete.")


def create_dockerfile(dockerfile_path: Path, vllm_source_dir_name: str, model_weights_dir_name: str, api_port: int):
    """
    Generates the Dockerfile for the CPU build.
    """
    logging.info(f"Creating Dockerfile at '{dockerfile_path}'")

    dockerfile_content = f"""
# Stage 1: Build vLLM for CPU
FROM {DOCKER_BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies, including python development headers and ninja
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    python3.11 \\
    python3.11-dev \\
    python3.11-venv \\
    python3-pip \\
    cmake \\
    build-essential \\
    ninja-build \\
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install numpy, which is needed by torch setup
RUN pip install --no-cache-dir numpy

# Install the specific PyTorch version required by vLLM for CPU
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Copy vLLM source code
COPY ./{vllm_source_dir_name} /app/{vllm_source_dir_name}
WORKDIR /app/{vllm_source_dir_name}

# Build and install vLLM for CPU
# It will use the pre-installed torch version and python dev headers
RUN VLLM_TARGET_DEVICE=cpu MAX_JOBS=$(nproc) pip install --no-cache-dir -e .

# Stage 2: Final Image
FROM {DOCKER_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.11 \\
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment with vLLM installed from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the downloaded model weights
COPY ./{model_weights_dir_name} /app/{model_weights_dir_name}

# Set up environment
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Expose the API port
EXPOSE {api_port}

# Start the vLLM OpenAI-compatible server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "/app/{model_weights_dir_name}", \\
     "--host", "0.0.0.0", \\
     "--port", "{str(api_port)}", \\
     "--tensor-parallel-size", "1", \\
     "--enforce-eager"]
"""
    dockerfile_path.write_text(dockerfile_content)


def clone_vllm(vllm_source_dir: Path):
    """
    Clones the vLLM repository if it doesn't already exist.
    """
    if vllm_source_dir.exists() and any(vllm_source_dir.iterdir()):
        logging.warning(
            f"vLLM source directory '{vllm_source_dir}' already exists. Skipping clone."
        )
        return
    logging.info(f"Cloning vLLM repository to '{vllm_source_dir}'...")
    run_command(["git", "clone", VLLM_REPO, str(vllm_source_dir)])


def send_test_request(port: int):
    """
    Sends a sample request to the running API server to verify it's working.
    """
    import openai

    logging.info("Sending test request to the API server...")
    client = openai.OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-needed",
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": "What is the primary purpose of the DeepSeek-OCR model?",
                }
            ],
        )
        response_content = completion.choices[0].message.content
        logging.info("--- Test Response ---")
        logging.info(response_content)
        logging.info("---")
        logging.info("Test request successful! The server is running correctly.")
    except Exception as e:
        logging.error(f"Test request failed: {e}")
        logging.error(
            "The container might be running, but the server failed to start. "
            "Check container logs with 'docker logs deepseek-ocr-container'"
        )
        raise


def main():
    """
    Main function to orchestrate the deployment pipeline.
    """
    parser = argparse.ArgumentParser(
        description=f"Deploy DeepSeek-OCR with vLLM on CPU using Docker. Version: {SCRIPT_VERSION}"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./model_data",
        help="Directory to store model weights and build artifacts.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to expose the API on the host machine.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="deepseek-ocr-vllm:latest",
        help="Name and tag for the Docker image.",
    )
    args = parser.parse_args()

    logging.info(f"--- DeepSeek-OCR Deployment Script v{SCRIPT_VERSION} ---")

    try:
        # Define paths relative to the model directory
        base_dir = Path(args.model_dir).resolve()
        base_dir.mkdir(exist_ok=True)

        model_weights_dir = base_dir / "model_weights"
        vllm_source_dir = base_dir / "vllm_source"
        dockerfile_path = base_dir / "Dockerfile"

        # 1. Download model
        download_model(model_weights_dir)

        # 2. Clone vLLM source
        clone_vllm(vllm_source_dir)

        # 3. Create Dockerfile
        create_dockerfile(
            dockerfile_path,
            vllm_source_dir.name,
            model_weights_dir.name,
            args.port
        )

        # 4. Build Docker image
        logging.info(f"Building Docker image '{args.image_name}'...")
        run_command(["docker", "build", "-t", args.image_name, "."], cwd=str(base_dir))
        logging.info("Docker image built successfully.")

        # 5. Deploy Docker container
        container_name = "deepseek-ocr-container"
        logging.info(f"Stopping and removing existing container named '{container_name}'...")
        # Stop and remove any previous container with the same name to avoid conflicts
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        subprocess.run(["docker", "rm", container_name], capture_output=True)

        logging.info(f"Starting Docker container '{container_name}'...")
        run_command([
            "docker", "run", "--rm", "-d",
            "-p", f"{args.port}:{args.port}",
            "--name", container_name,
            args.image_name
        ])
        logging.info(f"Container started. API should be available at http://localhost:{args.port}")

        # 6. Verify deployment with a test request
        logging.info("Waiting 30 seconds for the server to initialize...")
        time.sleep(30)
        send_test_request(args.port)

    except Exception as e:
        logging.error(f"Deployment pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

