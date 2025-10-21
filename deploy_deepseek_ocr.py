#!/usr/bin/env python3
import argparse
import base64
import logging
import os
import requests
import shutil
import subprocess
import sys
import time
from pathlib import Path

import docker
from huggingface_hub import snapshot_download
from openai import OpenAI, APIConnectionError

# --- Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

MODEL_ID = "deepseek-ai/DeepSeek-OCR"
VLLM_REPO_URL = "https://github.com/vllm-project/vllm.git"
TEST_IMAGE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-OCR/resolve/main/assets/show1.jpg"


# --- Helper Functions ---

def run_command(command, cwd=None):
    """Executes a shell command and raises an exception on failure."""
    logging.info(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, cwd=cwd, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        raise


def download_model(destination_dir: Path):
    """Downloads the DeepSeek-OCR model from Hugging Face."""
    logging.info(f"Downloading model '{MODEL_ID}' to '{destination_dir}'...")
    if destination_dir.exists():
        logging.warning(f"Destination directory '{destination_dir}' already exists. Skipping download.")
        return
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=destination_dir,
        local_dir_use_symlinks=False,
    )
    logging.info("Model download complete.")


def create_dockerfile(work_dir: Path, model_dir_name: str):
    """Programmatically creates the Dockerfile for the build."""
    dockerfile_content = f"""
# Base image: Ubuntu 22.04 LTS
FROM ubuntu:22.04

# Set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and build toolchain
RUN apt-get update -y && \\
    apt-get install -y --no-install-recommends \\
    git git-lfs python3.10-venv python3-pip python3-dev \\
    build-essential cmake ninja-build libnuma-dev \\
    gcc-12 g++-12 libtcmalloc-minimal4 && \\
    rm -rf /var/lib/apt/lists/*

# Set gcc-12 as the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \\
    --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Create a non-root user for security
RUN useradd -ms /bin/bash vllmuser
USER vllmuser
WORKDIR /home/vllmuser

# Copy vLLM source code
COPY --chown=vllmuser:vllmuser./vllm_source /home/vllmuser/vllm

# Create and activate a Python virtual environment
RUN python3 -m venv.venv
ENV PATH="/home/vllmuser/.venv/bin:$PATH"

# Install Python dependencies for CPU build
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r /home/vllmuser/vllm/requirements-cpu.txt \\
    --extra-index-url https://download.pytorch.org/whl/cpu

# Build and install vLLM for CPU
WORKDIR /home/vllmuser/vllm
RUN VLLM_TARGET_DEVICE=cpu python setup.py install

# Copy the model into the image
COPY --chown=vllmuser:vllmuser./{model_dir_name} /home/vllmuser/model

# Expose the internal API port
EXPOSE 8000

# Set the entrypoint to preload TCMalloc and start the server
CMD
"""
    dockerfile_path = work_dir / "Dockerfile"
    logging.info(f"Creating Dockerfile at '{dockerfile_path}'")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)


def build_docker_image(image_name: str, context_path: Path):
    """Clones vLLM and builds the Docker image."""
    vllm_source_path = context_path / "vllm_source"
    if not vllm_source_path.is_dir():
        logging.info(f"Cloning vLLM repository to '{vllm_source_path}'...")
        run_command()
    else:
        logging.info("vLLM repository already exists. Skipping clone.")

    logging.info(f"Building Docker image '{image_name}'...")
    client = docker.from_env()
    try:
        image, build_log = client.images.build(
            path=str(context_path),
            tag=image_name,
            rm=True,
            forcerm=True
        )
        for line in build_log:
            if 'stream' in line:
                print(line['stream'].strip())
        logging.info(f"Successfully built image: {image.tags}")
    except docker.errors.BuildError as e:
        logging.error("Docker build failed.")
        for line in e.build_log:
            if 'stream' in line:
                print(line['stream'].strip(), file=sys.stderr)
        raise


def deploy_container(image_name: str, container_name: str, port: int):
    """Deploys the Docker container."""
    logging.info(f"Deploying container '{container_name}' from image '{image_name}'...")
    client = docker.from_env()

    # Stop and remove existing container with the same name
    try:
        existing_container = client.containers.get(container_name)
        logging.warning(f"Found existing container '{container_name}'. Stopping and removing it.")
        existing_container.stop()
        existing_container.remove()
    except docker.errors.NotFound:
        pass  # No existing container, which is fine

    container = client.containers.run(
        image=image_name,
        name=container_name,
        detach=True,
        ports={'8000/tcp': port},
        shm_size='4g'  # Recommended for PyTorch
    )
    logging.info(f"Container '{container.name}' started with ID: {container.id}")
    return container


def run_inference_test(container_name: str, port: int):
    """Runs a test inference request against the deployed container."""
    logging.info("Running inference test...")
    api_base = f"http://localhost:{port}/v1"
    health_check_url = f"http://localhost:{port}/health"

    # Wait for the server to be ready
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    server_ready = False
    logging.info(f"Waiting for server to become healthy at {health_check_url}...")
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(health_check_url)
            if response.status_code == 200:
                logging.info("Server is healthy.")
                server_ready = True
                break
        except requests.ConnectionError:
            time.sleep(10)

    if not server_ready:
        logging.error("Server did not become healthy within the time limit.")
        raise RuntimeError("Server readiness check failed.")

    client = OpenAI(api_key="EMPTY", base_url=api_base)

    try:
        logging.info(f"Fetching test image from '{TEST_IMAGE_URL}'")
        response = requests.get(TEST_IMAGE_URL)
        response.raise_for_status()
        image_base64 = base64.b64encode(response.content).decode("utf-8")

        logging.info("Sending request to the model...")
        chat_completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}
                    ],
                }
            ],
            max_tokens=1024,
        )

        ocr_result = chat_completion.choices.message.content
        logging.info("--- OCR Result ---")
        print(ocr_result)
        logging.info("--------------------")

        if ocr_result and isinstance(ocr_result, str):
            logging.info("Inference test PASSED.")
        else:
            logging.error("Inference test FAILED: Received an empty or invalid response.")
            raise ValueError("Invalid response from model.")

    except Exception as e:
        logging.error(f"An error occurred during the inference test: {e}")
        raise


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Automated Deployment of DeepSeek-OCR on CPU with vLLM.")
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="./model_data",
        help="Destination directory for model download and build context."
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="deepseek-ocr-vllm-cpu:latest",
        help="Name and tag for the Docker image."
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="deepseek-ocr-server",
        help="Name for the running Docker container."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to expose the API server on the host machine."
    )
    args = parser.parse_args()

    work_dir = Path(args.dest_dir).resolve()
    work_dir.mkdir(exist_ok=True)

    model_dir_name = "model_weights"
    model_path = work_dir / model_dir_name

    try:
        # Step 1: Download the model
        download_model(model_path)

        # Step 2: Create the Dockerfile
        create_dockerfile(work_dir, model_dir_name)

        # Step 3: Build the Docker image
        build_docker_image(args.image_name, work_dir)

        # Step 4: Deploy the container
        container = deploy_container(args.image_name, args.container_name, args.port)

        # Step 5: Run the inference test
        run_inference_test(args.container_name, args.port)

        logging.info(
            f"\nDeployment successful! The DeepSeek-OCR server is running in container '{args.container_name}'.")
        logging.info(f"API Endpoint: http://localhost:{args.port}/v1")
        logging.info(f"To stop the server, run: docker stop {args.container_name}")

    except Exception as e:
        logging.error(f"Deployment pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()