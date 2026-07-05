# AIRE HPC Setup

University of Leeds AIRE cluster setup for running the synthetic VAWG conversation pipeline at scale.

---

## Why AIRE

Local generation runs (MacBook, Ollama, gpt-oss:20b) take roughly 30–60 seconds per LLM call. With three LLM calls per exchange (message generation, state assessment, completion check) a 40-turn conversation takes well over an hour locally. AIRE's GPU nodes (NVIDIA L40S, 48GB VRAM × 3 per node) reduce inference to seconds per call, making large-scale generation feasible.

AIRE also allows running multiple conversations in parallel as separate SLURM jobs — useful for generating a dataset across different seeds, character pairings, or scenarios.

---

## Running Ollama via Apptainer

Ollama is not available as a module on AIRE. The approved route for individual use is to run it inside an **Apptainer container**.

**What Apptainer is:** Apptainer (formerly Singularity) is the containerisation tool available on AIRE. It allows you to run software packaged as a container without requiring root access — which is why it is used on HPC systems instead of Docker. Containers built from Docker images can be converted to Apptainer's `.sif` format.

**Why a container for Ollama:** Ollama is a server process that manages model weights and serves inference via a local HTTP API. Packaging it in a container means it runs consistently on any AIRE GPU node regardless of what is installed on the host, and avoids any dependency conflicts with the system environment.

### Pulling the Ollama container

```bash
module load apptainer
cd $SCRATCH
apptainer pull ollama.sif docker://ollama/ollama:latest
```

This downloads the official Ollama Docker image from Docker Hub and converts it into a single `ollama.sif` file (~2.7GB) stored in scratch space. Only needs to be done once.

Model weights are stored separately in scratch space (not inside the container), so they persist across jobs and do not need to be re-downloaded each time.

---

## Python Environment

AIRE provides **Miniforge** as its Python distribution. A Conda environment is used rather than a plain Python venv because:

- Conda manages the Python version itself (system Python on HPC nodes is often old or locked)
- Miniforge is the available module — there is no standalone `python3` to rely on

### Setup

```bash
module load miniforge
conda create -n fyp python=3.11 -y
conda activate fyp
cd ~/synthetic-conversation-generation
pip install -e .
pip install ollama
```

`pip install -e .` installs the project as an editable package, meaning Python can resolve `import synthetic_conversation_generation` and any edits to `src/` take effect without reinstalling.

---

## File Layout on AIRE

```
~/                                        ← home directory (65GB, backed up)
  synthetic-conversation-generation/      ← code lives here

$SCRATCH/                                 ← scratch space (1TB, not backed up)
  ollama.sif                              ← Ollama Apptainer container
  models/                                 ← Ollama model weights
  conversations/                          ← generated conversation output
```

Large files (model weights, generated datasets) go in `$SCRATCH`. Code and configuration stay in `$HOME`.

---

## Transferring Code from Laptop

AIRE login nodes only accept inbound connections from wired campus connections. From off-campus, the **University of Leeds VPN** must be connected first.

```bash
# On laptop (VPN connected)
zip -r synthetic-conversation-generation.zip synthetic-conversation-generation \
  --exclude "*/\.venv/*" --exclude "*/__pycache__/*" --exclude "*.pyc"

scp synthetic-conversation-generation.zip sc23as2@aire.leeds.ac.uk:~

# On AIRE
cd ~
unzip synthetic-conversation-generation.zip
```

---

## SLURM Job Script

To be added once Ollama model pull and pipeline run are verified on a GPU node.
