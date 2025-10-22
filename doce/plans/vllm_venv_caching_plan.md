# vLLM venv caching and Docker build plan

Date: 2025-10-22
Repository: Deepseek-OCR-Docker
Location: `doce/plans/vllm_venv_caching_plan.md`

Purpose
- Capture the recommended strategy to speed up building the Python virtualenv and Python dependencies during Docker image builds for vLLM.
- Provide concrete Dockerfile snippets and build commands that use BuildKit cache mounts and a wheelhouse to dramatically reduce rebuild times.

Background
- The project runs an install script (see `command_vllm_install.txt`) inside the Docker build that creates a venv, builds vLLM from source, and installs dependencies.
- Because the Dockerfile copies large directories and then runs that install script as a single step, any change to source/model files invalidates the cached layer and forces a complete re-run of the expensive build.

Goals
- Make Python dependency installation (pip wheel / pip install) cacheable across Docker builds.
- Reduce iteration time when changing application code or model files that don't affect Python dependency compilation.
- Provide reproducible, repeatable builds that work well in local and CI environments.

High-level strategy
1. Separate concerns: copy only the minimum files required for dependency resolution (requirements files) and perform wheel building / pip install before copying the large or frequently-changing source and model files.
2. Prebuild wheels (wheelhouse) using `pip wheel --wheel-dir=/wheels` and install from `/wheels` with `pip install --no-index --find-links=/wheels`.
3. Use BuildKit cache mounts (`--mount=type=cache,target=/root/.cache/pip`) in Dockerfile RUN steps to keep pip caches between builds without baking caches into final image layers.
4. Persist build cache across separate docker builds and CI runs using `docker buildx --cache-to` / `--cache-from` (local folder or registry-backed cache).
5. Prefer an absolute, deterministic venv path (for example `/opt/venv`) created early in the Dockerfile so the venv layer is stable and isolated from later COPY steps.

Concrete Dockerfile pattern
- Insert into `model_data/Dockerfile` (adapt to repository layout):

```dockerfile
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libnuma-dev python3.11-venv python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Deterministic venv path
ENV VENV_PATH=/opt/venv
RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy only vLLM requirement files first to minimize cache busting
COPY ../third_party/vllm/requirements /tmp/vllm_requirements
COPY ./command_vllm_install.txt /tmp/command_vllm_install.txt

# Use BuildKit cache mount for pip cache and build wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel setuptools \
    && pip wheel --wheel-dir=/wheels -r /tmp/vllm_requirements/cpu.txt \
    && pip install --no-index --find-links=/wheels -r /tmp/vllm_requirements/cpu.txt

# Copy the rest (won't bust the pip/wheel layers above)
COPY ./model_weights /app/model_weights
COPY ./install_vllm.sh /app/install_vllm.sh
COPY ./run_model.sh /app/run_model.sh
COPY ../third_party/vllm /app/vllm_source

# Run the install script (should be faster since dependencies installed)
RUN bash /app/install_vllm.sh

EXPOSE 8000
CMD ["./run_model.sh"]
```

Build commands (local + persistent cache)

- Local cache folder (good for iterative local development):

```bash
export DOCKER_BUILDKIT=1
# Build and persist cache to ./.buildx-cache
docker buildx build \
  --progress=plain \
  --tag deepseek-ocr:latest \
  --cache-to=type=local,dest=./.buildx-cache \
  --cache-from=type=local,src=./.buildx-cache \
  -f model_data/Dockerfile .
```

- Registry-backed cache (CI-friendly):

```bash
docker buildx build \
  --cache-to=type=registry,ref=yourrepo/deepseek-buildcache:latest \
  --cache-from=type=registry,ref=yourrepo/deepseek-buildcache:latest \
  --tag yourrepo/deepseek-image:latest \
  -f model_data/Dockerfile .
```

Notes about `command_vllm_install.txt` / install script
- Move pure Python dependency installs to the requirements files so the Dockerfile can wheel/install them with caching. Keep only source build logic (like `python -m build --wheel`) in the install script if necessary.
- Consider making the script idempotent and able to detect if installation steps are already satisfied (skip rebuild if wheels exist or package already installed).
- If `install_vllm.sh` still needs to build the vLLM wheel from source, do that after dependencies are installed. Consider caching the `dist/*.whl` result in a build cache location if it helps.

Other recommendations
- Generate and commit an explicit, pinned `requirements.txt` (use `pip-compile` or freeze) for reproducible builds.
- If builds are still slow because vLLM compiles heavy native code, create a base image that already includes the built wheel(s) and use that as FROM in downstream images.
- For advanced speedups, use prebuilt manylinux wheels for heavy deps or a private PyPI index that holds compiled wheels.

Next steps (pick one)
- Patch `model_data/Dockerfile` to apply the wheelhouse + BuildKit pattern (I can create a safe patch).  
- Patch `command_vllm_install.txt` / `install_vllm.sh` to split dependency install from source build.  
- Create a GitHub Actions workflow to persist buildx cache between CI builds.

Status
- This plan file saved in `doce/plans/vllm_venv_caching_plan.md`.
- Suggested follow-ups: implement the Dockerfile patch and test a build using `docker buildx` with the cache options above.


```yaml
# Example minimal CI hint (GitHub Actions) - idea sketch
# - Uses buildx/cache action to persist cache
# - Runs docker buildx build with cache-from/cache-to
```

