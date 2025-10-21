import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# --- Configuration ---import argparse
# import logging
# import os
# import subprocess
# import sys
# import time
# from pathlib import Path
#
# # --- Configuration ---
# SCRIPT_VERSION = "3.2 (Final)"
# MODEL_ID = "deepseek-ai/DeepSeek-OCR"
# VLLM_REPO = "https://github.com/vllm-project/vllm.git"
# DOCKER_BASE_IMAGE = "python:3.11-slim"
# CONTAINER_NAME = "deepseek-ocr-container"
# TROUBLESHOOT_IMAGE_NAME = "deepseek-ocr-vllm:troubleshoot"
#
# # --- Logging Setup ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     stream=sys.stdout,
# )
#
# def run_command(command, cwd=None, capture_output=False, text=True, raise_on_error=True):
#     """
#     Executes a shell command.
#     """
#     if isinstance(command, str):
#         command = command.split()
#
#     logging.info(f"Running command: {' '.join(command)}" + (f" in {cwd}" if cwd else ""))
#     try:
#         result = subprocess.run(
#             command,
#             check=raise_on_error,
#             cwd=cwd,
#             capture_output=capture_output,
#             text=text
#         )
#         if capture_output and result.stdout:
#             logging.info(f"Command output:\n{result.stdout}")
#         return result
#     except subprocess.CalledProcessError as e:
#         logging.error(f"Command failed with exit code {e.returncode}")
#         if e.stdout:
#             logging.error(f"STDOUT: {e.stdout}")
#         if e.stderr:
#             logging.error(f"STDERR: {e.stderr}")
#         if raise_on_error:
#             raise
#
# def download_model(model_weights_dir: Path):
#     """
#     Downloads the model from Hugging Face if it doesn't already exist.
#     """
#     if model_weights_dir.exists() and any(model_weights_dir.iterdir()):
#         logging.warning(
#             f"Destination directory '{model_weights_dir}' already exists and is not empty. Skipping download."
#         )
#         return
#
#     logging.info(f"Downloading model '{MODEL_ID}' to '{model_weights_dir}'...")
#     from huggingface_hub import snapshot_download
#     from tqdm import tqdm
#
#     snapshot_download(
#         MODEL_ID,
#         local_dir=str(model_weights_dir),
#         local_dir_use_symlinks=False,
#         tqdm_class=tqdm,
#     )
#     logging.info("Model download complete.")
#
# def create_dockerfile(dockerfile_path: Path, vllm_source_dir_name: str, model_weights_dir_name: str, api_port: int):
#     """
#     Generates the definitive Dockerfile with separated pip install commands.
#     """
#     logging.info(f"Creating Dockerfile at '{dockerfile_path}' (Version: {SCRIPT_VERSION})")
#
#     dockerfile_content = f"""
# # Stage 1: Builder
# FROM {DOCKER_BASE_IMAGE} AS builder
#
# ENV DEBIAN_FRONTEND=noninteractive
#
# # Install all necessary system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \\
#     git \\
#     build-essential \\
#     libnuma-dev \\
#     && rm -rf /var/lib/apt/lists/*
#
# WORKDIR /app
#
# # Upgrade pip and install the 'build' package first
# RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir build
#
# # FINAL FIX: Install torch/torchvision from their special index first
# RUN pip install --no-cache-dir \\
#     "torch==2.8.0+cpu" \\
#     "torchvision==0.19.0+cpu" \\
#     --extra-index-url https://download.pytorch.org/whl/cpu
#
# # Now, install transformers from the standard PyPI
# RUN pip install --no-cache-dir "transformers==4.49.2"
#
# # Copy vLLM source code
# COPY ./{vllm_source_dir_name} .
#
# # Build the vLLM wheel. It will use the pre-installed libraries.
# RUN VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
#
# # Stage 2: Final Production Image
# FROM {DOCKER_BASE_IMAGE}
#
# WORKDIR /app
#
# # Copy the installed Python packages from the builder stage
# COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
#
# # Copy the vLLM wheel
# COPY --from=builder /app/dist/*.whl /app/wheel/
#
# # Install the built vLLM wheel
# RUN pip install --no-cache-dir /app/wheel/*.whl
#
# # Copy the downloaded model weights
# COPY ./{model_weights_dir_name} /app/model_weights
#
# EXPOSE {api_port}
#
# # Run the API server
# CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \\
#      "--model", "/app/model_weights", \\
#      "--host", "0.0.0.0", \\
#      "--port", "{str(api_port)}", \\
#      "--tensor-parallel-size", "1", \\
#      "--enforce-eager"]
# """
#     dockerfile_path.write_text(dockerfile_content)
#
# def clone_vllm(vllm_source_dir: Path):
#     """
#     Clones the vLLM repository if it doesn't already exist.
#     """
#     if vllm_source_dir.exists() and any(vllm_source_dir.iterdir()):
#         logging.warning(
#             f"vLLM source directory '{vllm_source_dir}' already exists. Skipping clone."
#         )
#         return
#     logging.info(f"Cloning vLLM repository to '{vllm_source_dir}'...")
#     run_command(["git", "clone", VLLM_REPO, str(vllm_source_dir)])
#
# def send_test_request(port: int):
#     """
#     Sends a sample request to the running API server to verify it's working.
#     """
#     import openai
#
#     logging.info("Sending test request to the API server...")
#     client = openai.OpenAI(
#         base_url=f"http://localhost:{port}/v1",
#         api_key="not-needed",
#     )
#     try:
#         completion = client.chat.completions.create(
#             model=MODEL_ID,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": "What is the primary purpose of the DeepSeek-OCR model?",
#                 }
#             ],
#         )
#         response_content = completion.choices[0].message.content
#         logging.info("--- Test Response ---")
#         logging.info(response_content)
#         logging.info("---")
#         logging.info("Test request successful! The server is running correctly.")
#     except Exception as e:
#         logging.error(f"Test request failed: {e}")
#         logging.error(
#             "The container might be running, but the server failed to start. "
#             f"Check container logs with 'docker logs {CONTAINER_NAME}'"
#         )
#         raise
#
# def main():
#     """
#     Main function to orchestrate the deployment pipeline.
#     """
#     parser = argparse.ArgumentParser(
#         description=f"Deploy DeepSeek-OCR with vLLM on CPU using Docker. Version: {SCRIPT_VERSION}"
#     )
#     parser.add_argument(
#         "--model-dir",
#         type=str,
#         default="./model_data",
#         help="Directory to store model weights and build artifacts.",
#     )
#     parser.add_argument(
#         "--port",
#         type=int,
#         default=8000,
#         help="Port to expose the API on the host machine.",
#     )
#     parser.add_argument(
#         "--image-name",
#         type=str,
#         default="deepseek-ocr-vllm:latest",
#         help="Name and tag for the Docker image.",
#     )
#     parser.add_argument(
#         "--troubleshoot",
#         action="store_true",
#         help="If build fails, tag the last successful layer for debugging.",
#     )
#     args = parser.parse_args()
#
#     logging.info(f"--- DeepSeek-OCR Deployment Script v{SCRIPT_VERSION} ---")
#
#     try:
#         base_dir = Path(args.model_dir).resolve()
#         base_dir.mkdir(exist_ok=True)
#
#         model_weights_dir = base_dir / "model_weights"
#         vllm_source_dir = base_dir / "vllm_source"
#         dockerfile_path = base_dir / "Dockerfile"
#
#         download_model(model_weights_dir)
#         clone_vllm(vllm_source_dir)
#         create_dockerfile(
#             dockerfile_path,
#             vllm_source_dir.name,
#             model_weights_dir.name,
#             args.port
#         )
#
#         logging.info(f"Building Docker image '{args.image_name}'...")
#         build_command = ["docker", "build", "-t", args.image_name, "."]
#
#         should_raise_on_error = not args.troubleshoot
#         build_process = subprocess.run(
#             build_command,
#             cwd=str(base_dir),
#             capture_output=True,
#             text=True
#         )
#
#         if build_process.returncode != 0:
#             logging.error("Docker build failed.")
#             print("--- DOCKER BUILD STDOUT ---")
#             print(build_process.stdout)
#             print("--- DOCKER BUILD STDERR ---")
#             print(build_process.stderr)
#
#             if args.troubleshoot:
#                 logging.warning("Entering troubleshoot mode.")
#                 dangling_images_result = run_command(
#                     'docker images -q --filter dangling=true',
#                     capture_output=True,
#                     raise_on_error=False
#                 )
#                 image_id = dangling_images_result.stdout.strip().split('\n')[0] if dangling_images_result.stdout else None
#                 if image_id:
#                     logging.info(f"Found last successful layer with ID: {image_id}")
#                     run_command(f"docker tag {image_id} {TROUBLESHOOT_IMAGE_NAME}")
#                     logging.info(f"Successfully tagged as '{TROUBLESHOOT_IMAGE_NAME}'")
#                     print("\n" + "="*80)
#                     print("TROUBLESHOOTING SHELL READY")
#                     print(f"To debug the build environment, run the following command:")
#                     print(f"  docker run -it --entrypoint /bin/bash {TROUBLESHOOT_IMAGE_NAME}")
#                     print("="*80 + "\n")
#                 else:
#                     logging.error("Could not find a dangling image to tag for troubleshooting.")
#             sys.exit(1)
#
#         logging.info("Docker image built successfully.")
#
#         logging.info(f"Stopping and removing existing container named '{CONTAINER_NAME}'...")
#         subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, text=True)
#         subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True, text=True)
#
#         logging.info(f"Starting Docker container '{CONTAINER_NAME}'...")
#         run_command([
#             "docker", "run", "--rm", "-d",
#             "-p", f"{args.port}:{args.port}",
#             "--name", CONTAINER_NAME,
#             args.image_name
#         ])
#         logging.info(f"Container started. API should be available at http://localhost:{args.port}")
#
#         logging.info("Waiting 30 seconds for the server to initialize...")
#         time.sleep(30)
#         send_test_request(port=args.port)
#
#     except Exception as e:
#         logging.error(f"Deployment pipeline failed: {e}")
#         sys.exit(1)
#
# if __name__ == "__main__":
#     main()
SCRIPT_VERSION = "3.1 (Final, Runtime Fix)"
MODEL_ID = "deepseek-ai/DeepSeek-OCR"
VLLM_REPO = "https://github.com/vllm-project/vllm.git"
DOCKER_BASE_IMAGE = "python:3.11-slim"
CONTAINER_NAME = "deepseek-ocr-container"
TROUBLESHOOT_IMAGE_NAME = "deepseek-ocr-vllm:troubleshoot"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def run_command(command, cwd=None, capture_output=False, text=True, raise_on_error=True):
    """
    Executes a shell command.
    """
    if isinstance(command, str):
        command = command.split()

    logging.info(f"Running command: {' '.join(command)}" + (f" in {cwd}" if cwd else ""))
    try:
        result = subprocess.run(
            command,
            check=raise_on_error,
            cwd=cwd,
            capture_output=capture_output,
            text=text
        )
        if capture_output and result.stdout:
            logging.info(f"Command output:\n{result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logging.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr}")
        if raise_on_error:
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
    Generates the definitive Dockerfile with an explicit, compatible torch/torchvision install.
    """
    logging.info(f"Creating Dockerfile at '{dockerfile_path}' (Version: {SCRIPT_VERSION})")

    dockerfile_content = f"""
# Stage 1: Builder
FROM {DOCKER_BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    build-essential \\
    libnuma-dev \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install the 'build' package
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir build

# CRITICAL RUNTIME FIX: Install compatible torch and torchvision together for CPU
# Also install transformers, which is a key dependency for the model.
RUN pip install --no-cache-dir \\
    "torch==2.8.0+cpu" \\
    "torchvision==0.19.0+cpu" \\
    "transformers==4.49.2" \\
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy vLLM source code
COPY ./{vllm_source_dir_name} .

# Build the vLLM wheel. It will use the pre-installed libraries.
RUN VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation

# Stage 2: Final Production Image
FROM {DOCKER_BASE_IMAGE}

WORKDIR /app

# Copy the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the vLLM wheel
COPY --from=builder /app/dist/*.whl /app/wheel/

# Install the built vLLM wheel
RUN pip install --no-cache-dir /app/wheel/*.whl

# Copy the downloaded model weights
COPY ./{model_weights_dir_name} /app/model_weights

EXPOSE {api_port}

# Run the API server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \\
     "--model", "/app/model_weights", \\
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
            f"Check container logs with 'docker logs {CONTAINER_NAME}'"
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
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        help="If build fails, tag the last successful layer for debugging.",
    )
    args = parser.parse_args()

    logging.info(f"--- DeepSeek-OCR Deployment Script v{SCRIPT_VERSION} ---")

    try:
        base_dir = Path(args.model_dir).resolve()
        base_dir.mkdir(exist_ok=True)

        model_weights_dir = base_dir / "model_weights"
        vllm_source_dir = base_dir / "vllm_source"
        dockerfile_path = base_dir / "Dockerfile"

        download_model(model_weights_dir)
        clone_vllm(vllm_source_dir)
        create_dockerfile(
            dockerfile_path,
            vllm_source_dir.name,
            model_weights_dir.name,
            args.port
        )

        logging.info(f"Building Docker image '{args.image_name}'...")
        build_command = ["docker", "build", "-t", args.image_name, "."]

        should_raise_on_error = not args.troubleshoot
        build_process = subprocess.run(
            build_command,
            cwd=str(base_dir),
            capture_output=True,
            text=True
        )

        if build_process.returncode != 0:
            logging.error("Docker build failed.")
            print("--- DOCKER BUILD STDOUT ---")
            print(build_process.stdout)
            print("--- DOCKER BUILD STDERR ---")
            print(build_process.stderr)

            if args.troubleshoot:
                logging.warning("Entering troubleshoot mode.")
                dangling_images_result = run_command(
                    'docker images -q --filter dangling=true',
                    capture_output=True,
                    raise_on_error=False
                )
                image_id = dangling_images_result.stdout.strip().split('\n')[
                    0] if dangling_images_result.stdout else None
                if image_id:
                    logging.info(f"Found last successful layer with ID: {image_id}")
                    run_command(f"docker tag {image_id} {TROUBLESHOOT_IMAGE_NAME}")
                    logging.info(f"Successfully tagged as '{TROUBLESHOOT_IMAGE_NAME}'")
                    print("\n" + "=" * 80)
                    print("TROUBLESHOOTING SHELL READY")
                    print(f"To debug the build environment, run the following command:")
                    print(f"  docker run -it --entrypoint /bin/bash {TROUBLESHOOT_IMAGE_NAME}")
                    print("=" * 80 + "\n")
                else:
                    logging.error("Could not find a dangling image to tag for troubleshooting.")
            sys.exit(1)

        logging.info("Docker image built successfully.")

        logging.info(f"Stopping and removing existing container named '{CONTAINER_NAME}'...")
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, text=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True, text=True)

        logging.info(f"Starting Docker container '{CONTAINER_NAME}'...")
        run_command([
            "docker", "run", "--rm", "-d",
            "-p", f"{args.port}:{args.port}",
            "--name", CONTAINER_NAME,
            args.image_name
        ])
        logging.info(f"Container started. API should be available at http://localhost:{args.port}")

        logging.info("Waiting 30 seconds for the server to initialize...")
        time.sleep(30)
        send_test_request(port=args.port)

    except Exception as e:
        logging.error(f"Deployment pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

