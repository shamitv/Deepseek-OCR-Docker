How to Deploy DeepSeek-OCR with vLLM on a CPU Server

This guide provides step-by-step instructions to download, build, and deploy the deepseek-ai/DeepSeek-OCR model on a CPU-only Ubuntu machine using a Python script. The script automates the process of setting up a vLLM server within a Docker container.

Prerequisites

Before you begin, ensure you have the following installed on your Ubuntu machine:

Docker: The script requires Docker to build and run the model container.  You can find installation instructions here: Install Docker Engine on Ubuntu.

Python 3.10+ and pip.

Git: For cloning the vLLM repository.

Setup

Save the Files:
Save the deploy_deepseek_ocr.py and requirements.txt files to a new directory on your machine.

mkdir deepseek-deployment
cd deepseek-deployment
# Now, place the two files in this directory


Install Python Dependencies:
Create a virtual environment and install the required Python packages from requirements.txt.

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Deployment

The deploy_deepseek_ocr.py script handles the entire deployment pipeline.

Run the Deployment Script:
Execute the script from your terminal. You can customize the model directory and the API port.

python deploy_deepseek_ocr.py --port 12000 --model-dir ./model_data


Command-Line Arguments:

--model-dir DIRECTORY: Specifies the directory to download the model and store build artifacts. Default: ./model_data. The script is idempotent; if this directory exists, it will skip downloading the model and cloning vLLM.

--port PORT: The external port to expose for the OpenAI-compatible API. Default: 8000. Port 12000 is perfectly valid.

--image-name NAME: The name for the Docker image to be built. Default: deepseek-ocr-vllm:latest.

What the Script Does:

Logs its version number for easy tracking.

Downloads the deepseek-ai/DeepSeek-OCR model from Hugging Face (if not already present).

Clones the vLLM repository (if not already present).

Generates a Dockerfile tailored for a CPU build.

Builds a Docker image containing PyTorch (CPU), vLLM, and the model weights.

Starts a Docker container, running the vLLM API server on the specified port.

Waits for the server to become active and sends a test request to confirm it's working.

Verification and Interaction

Once the script completes successfully, you will see a "Test request successful!" message with the model's response. The API server is now running and accessible.

You can interact with it using any OpenAI-compatible client library or curl.

Example using curl:

curl http://localhost:12000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'


FAQ & Further Information

If I run the build 5 times, will it download the model 5 times?
No. The script checks if the destination directory (--model-dir) already exists. If it does, it skips downloading the model and cloning vLLM, making subsequent runs much faster.

Does GCC 13 work?
Yes, vLLM and its dependencies like PyTorch are compatible with modern compilers like GCC 13. The Docker image created by the script uses the stable toolchain from Ubuntu 22.04 repositories, which is well-tested.

Exploring llama.cpp as an alternative:
llama.cpp is an excellent, highly-optimized inference engine for running LLMs on CPUs. It's a great alternative if you want to explore different performance characteristics. However, it has its own server implementation and build process, which is different from the vLLM approach used in this script.

How to stop the server:
Find the container ID and stop it.

docker ps
docker stop <container_id>
