import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
SCRIPT_VERSION = "4.0 (Shell-Based)"
DOCKER_BASE_IMAGE = "python:3.11-slim"
CONTAINER_NAME = "deepseek-ocr-container"
TROUBLESHOOT_IMAGE_NAME = "deepseek-ocr-vllm:troubleshoot"
INSTALL_COMMANDS_FILE = "command_vllm_install.txt"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def run_command(command, cwd=None, raise_on_error=True):
    """Executes a shell command, streaming its output."""
    if isinstance(command, str):
        command = command.split()
    logging.info(f"Running command: {' '.join(command)}" + (f" in {cwd}" if cwd else ""))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in iter(process.stdout.readline, ''):
        sys.stdout.write(line)
    process.wait()
    if raise_on_error and process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def create_install_script(source_txt_path: Path, output_sh_path: Path):
    """Reads commands from a text file and creates an executable shell script."""
    logging.info(f"Generating install script at '{output_sh_path}' from '{source_txt_path}'...")
    if not source_txt_path.exists():
        logging.error(f"FATAL: Installation command file not found at '{source_txt_path}'")
        sys.exit(1)

    with open(source_txt_path, 'r') as f:
        commands = f.read()

    script_content = f"""#!/bin/bash
# Auto-generated from {source_txt_path.name} by deploy_deepseek_ocr.py
set -e
set -x

{commands}

echo "Installation script completed successfully."
"""
    output_sh_path.write_text(script_content)
    output_sh_path.chmod(0o755)  # Make the script executable
    logging.info("Install script created successfully.")


def create_run_script(output_sh_path: Path, model_dir_name: str, port: int):
    """Creates the script to run the vLLM server."""
    logging.info(f"Generating run script at '{output_sh_path}'...")
    script_content = f"""#!/bin/bash
set -x

# Activate the virtual environment created during installation
source /app/vllm_source/myenv/bin/activate

# Start the server
python -m vllm.entrypoints.openai.api_server \\
    --model "/app/{model_dir_name}" \\
    --host "0.0.0.0" \\
    --port "{port}" \\
    --tensor-parallel-size 1 \\
    --enforce-eager
"""
    output_sh_path.write_text(script_content)
    output_sh_path.chmod(0o755)  # Make the script executable
    logging.info("Run script created successfully.")


def create_dockerfile(dockerfile_path: Path, model_dir_name: str, install_script_name: str, run_script_name: str):
    """Generates a simplified Dockerfile that executes the install script."""
    logging.info(f"Creating Dockerfile at '{dockerfile_path}'...")
    dockerfile_content = f"""
FROM {DOCKER_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies required by the install script
RUN apt-get update && apt-get install -y --no-install-recommends \\
    git \\
    python3.11-venv \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the model files first
COPY ./{model_dir_name} /app/{model_dir_name}

# Copy the generated install and run scripts
COPY ./{install_script_name} /app/{install_script_name}
COPY ./{run_script_name} /app/{run_script_name}

# Run the installation script
# This is now the main build step
RUN ./{install_script_name}

# Expose the port and set the run script as the default command
EXPOSE 8000
CMD ["./run_model.sh"]
"""
    dockerfile_path.write_text(dockerfile_content)
    logging.info("Dockerfile created successfully.")


def main():
    parser = argparse.ArgumentParser(
        description=f"Deploy DeepSeek-OCR with vLLM on CPU using Docker. Version: {SCRIPT_VERSION}"
    )
    parser.add_argument("--model-dir", type=str, default="./model_data",
                        help="Directory for model weights and build artifacts.")
    parser.add_argument("--port", type=int, default=8000, help="API port on the host.")
    parser.add_argument("--image-name", type=str, default="deepseek-ocr-vllm:latest", help="Name for the Docker image.")
    parser.add_argument("--troubleshoot", action="store_true", help="If build fails, tag the last layer for debugging.")
    args = parser.parse_args()

    logging.info(f"--- DeepSeek-OCR Deployment Scripter v{SCRIPT_VERSION} ---")

    base_dir = Path(args.model_dir).resolve()
    base_dir.mkdir(exist_ok=True)

    install_commands_path = base_dir / INSTALL_COMMANDS_FILE
    install_script_path = base_dir / "install_vllm.sh"
    run_script_path = base_dir / "run_model.sh"
    dockerfile_path = base_dir / "Dockerfile"
    model_weights_dir = base_dir / "model_weights"  # Assuming model weights are here

    try:
        # Step 1: Generate scripts from templates/commands
        create_install_script(install_commands_path, install_script_path)
        create_run_script(run_script_path, model_weights_dir.name, args.port)
        create_dockerfile(dockerfile_path, model_weights_dir.name, install_script_path.name, run_script_path.name)

        # Step 2: Build the Docker image
        logging.info(f"Building Docker image '{args.image_name}'...")
        build_command = ["docker", "build", "--progress=plain", "-t", args.image_name, "."]

        try:
            run_command(build_command, cwd=str(base_dir))
            logging.info("Docker image built successfully.")
        except subprocess.CalledProcessError:
            logging.error("Docker build failed.")
            if args.troubleshoot:
                logging.warning("Entering troubleshoot mode.")
                # Find the most recent dangling image (the last successful layer)
                result = subprocess.run('docker images -q --filter dangling=true', capture_output=True, text=True,
                                        shell=True)
                image_id = result.stdout.strip().split('\n')[0] if result.stdout else None
                if image_id:
                    logging.info(f"Found last successful layer with ID: {image_id}")
                    run_command(f"docker tag {image_id} {TROUBLESHOOT_IMAGE_NAME}")
                    print("\n" + "=" * 80)
                    print("TROUBLESHOOTING SHELL READY")
                    print(f"To debug, run this command to get a shell inside the container:")
                    print(f"  docker run -it --entrypoint /bin/bash {TROUBLESHOOT_IMAGE_NAME}")
                    print("\nThen, inside the shell, you can manually run the install script:")
                    print("  ./install_vllm.sh")
                    print("=" * 80 + "\n")
                else:
                    logging.error("Could not find a dangling image to tag for troubleshooting.")
            sys.exit(1)

        # Step 3: Deploy the container
        logging.info(f"Stopping and removing any existing container named '{CONTAINER_NAME}'...")
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, text=True)
        subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True, text=True)

        logging.info(f"Starting new container '{CONTAINER_NAME}'...")
        run_command([
            "docker", "run", "-d", "-p", f"{args.port}:{args.port}",
            "--name", CONTAINER_NAME, args.image_name
        ], raise_on_error=False)  # Don't fail script if container exits immediately

        logging.info("Deployment command sent. Waiting 10 seconds to check container status...")
        time.sleep(10)

        # Check if the container is still running
        result = subprocess.run(f"docker ps -q --filter name={CONTAINER_NAME}", capture_output=True, text=True,
                                shell=True)
        if not result.stdout.strip():
            logging.error("Container failed to start or exited immediately.")
            logging.error("Check the container logs for the full error traceback:")
            print(f"\n  docker logs {CONTAINER_NAME}\n")
            sys.exit(1)

        logging.info(f"Container is running. API should be available at http://localhost:{args.port}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

