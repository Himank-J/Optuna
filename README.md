# Lightning Template Hydra with Optuna

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2Bcpu-orange.svg)
![Hydra](https://img.shields.io/badge/Hydra-1.3.2-green.svg)
![Optuna](https://img.shields.io/badge/Optuna-3.25.0-purple.svg)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Output](#output)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training with Hyperparameter Optimization](#training-with-hyperparameter-optimization)
  - [Generating Hyperparameter Reports](#generating-hyperparameter-reports)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Continuous Integration](#continuous-integration)
- [Docker Support](#docker-support)
- [Testing](#testing)
- [License](#license)

## Introduction

Welcome to the **Lightning Template Hydra with Optuna** repository! This project leverages [PyTorch Lightning](https://www.pytorchlightning.ai/) for streamlined deep learning workflows, [Hydra](https://hydra.cc/) for flexible configuration management, and [Optuna](https://optuna.org/) for efficient hyperparameter optimization. The primary focus is on building and optimizing a Dog Breed Classifier using these powerful tools.

## Features

- **Modular Configuration**: Easily manage and override configurations using Hydra.
- **Hyperparameter Optimization**: Utilize Optuna for automated and efficient hyperparameter tuning.
- **PyTorch Lightning Integration**: Simplify training loops and leverage Lightning's advanced features.
- **Automatic Report Generation**: Generate detailed hyperparameter optimization reports with metrics and visualizations.
- **Continuous Integration**: Automated workflows using GitHub Actions for training and reporting.
- **Docker Support**: Containerize the application for consistent environments across different systems.
- **Comprehensive Testing**: Ensure code reliability with extensive pytest suites.

## Architecture

The repository is structured to promote modularity and scalability. Below is an overview of the key components:

- **`src/`**: Contains the source code.
  - **`models/`**: Model definitions.
  - **`datamodules/`**: Data handling modules.
  - **`utils/`**: Utility functions, including logging.
  - **`train.py`**: Script to train the model.
  - **`eval.py`**: Script to evaluate the model.
  - **`infer.py`**: Script for making predictions on new data.
  - **`scripts/generate_hparam_report.py`**: Script to generate hyperparameter optimization reports.
- **`configs/`**: Hydra configuration files.
  - **`train.yaml`**, **`eval.yaml`**, **`infer.yaml`**: Main configurations for different stages.
  - **`hparams_search/optuna.yaml`**: Configuration for Optuna hyperparameter optimization.
  - **`callbacks/`**, **`logger/`**, **`trainer/`**, **`data/`**, **`model/`**, **`experiment/`**: Sub-configurations.
- **`tests/`**: Pytest suites for various components.
- **`Dockerfile`**: Docker configuration for containerizing the application.
- **`.github/workflows/hparam_report.yml`**: GitHub Actions workflow for automated training and reporting.
- **`README.md`**: This documentation file.

## Output

The hparam_report generates below report in github comments

**Hyperparameters Benchmarking across Different runs**
<img width="882" alt="image" src="https://github.com/user-attachments/assets/614bc16b-d556-44b1-b32d-4b136c1e3eb9">

**Best Hyperparameters**
<img width="557" alt="image" src="https://github.com/user-attachments/assets/28b82b9e-cc64-4514-a0d7-3311c8f1c697">

**Plots**
<img width="817" alt="image" src="https://github.com/user-attachments/assets/c742f120-ae72-4a2e-a0b6-bf69410166a5">

<img width="783" alt="image" src="https://github.com/user-attachments/assets/565d5a23-b0d1-470b-bd1e-19d07d7dba3a">

## Installation

### Prerequisites

- **Python 3.12**: Ensure you have Python installed. You can download it [here](https://www.python.org/downloads/).
- **uv**: A versatile package manager for Python projects.
- **Docker** (optional): For containerization.

### Clone the Repository

```bash
git clone https://github.com/your-username/lightning-template-hydra.git
cd lightning-template-hydra
```

### Using `uv` for Dependency Management

The project uses `uv` for managing dependencies and environments.

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

### Using Docker

Alternatively, you can build and run the project using Docker.

```bash
# Build the Docker image
docker build -t lightning-template-hydra .

# Run a container from the image
docker run -it --rm lightning-template-hydra
```

## Configuration

Configurations are managed using Hydra, allowing for easy overrides and modular setups. All configurations are located in the `configs/` directory.

### Key Configuration Files

- **`configs/train.yaml`**: Configuration for the training process.
- **`configs/hparams_search/optuna.yaml`**: Configuration for Optuna-based hyperparameter search.
- **`configs/model/classifier.yaml`**: Model-specific configurations.
- **`configs/data/dogbreed.yaml`**: Data module configurations.
- **`configs/callbacks/`**: Callback configurations like ModelCheckpoint, EarlyStopping, etc.
- **`configs/logger/`**: Logger configurations (e.g., TensorBoard, CSVLogger).
- **`configs/trainer/`**: PyTorch Lightning Trainer configurations.

### Overriding Configurations

You can override configurations directly via the command line. For example, to change the batch size during training:

```bash
python train.py data.batch_size=64
```

## Usage

### Training with Hyperparameter Optimization

The training process integrates Optuna for hyperparameter optimization. Hyperparameters are defined in the Optuna configuration file.

#### Running Training with Hydra and Optuna

```bash
python train.py hparams_search=optuna --multirun
```

This command initiates a hyperparameter optimization sweep using Optuna as the sweeper and runs multiple training experiments in parallel.

### Generating Hyperparameter Reports

After training, you can generate detailed reports summarizing the hyperparameter search results.

#### Generating the Report

```bash
python scripts/generate_hparam_report.py
```

This script processes the training logs and outputs a `hparam_report.md` file containing tables and plots of the hyperparameter performance.

### Evaluation

Evaluate the trained model using the evaluation script.

#### Running Evaluation

```bash
python eval.py
```

The evaluation script loads the best checkpoint and computes metrics on the test dataset, saving the results to `results.json`.

### Inference

Make predictions on new images using the inference script.

#### Running Inference

```bash
python infer.py
```

This script processes images from the specified input directory, generates predictions, and saves annotated images with predictions to the output directory.

## Continuous Integration

The project utilizes GitHub Actions for automated workflows, including training and report generation upon pushing changes to the `main` branch.

### GitHub Actions Workflow

- **File**: `.github/workflows/hparam_report.yml`

#### Workflow Steps

1. **Checkout Repository**: Uses `actions/checkout@v4`.
2. **Set Up CML**: Uses `iterative/setup-cml@v2`.
3. **Install `uv`**: Uses `astral-sh/setup-uv@v2`.
4. **Set Up Python**: Installs Python 3.12.
5. **Install Dependencies**: Syncs dependencies using `uv`.
6. **Run Hyperparameter Optimization**: Executes the training script with Optuna.
7. **Generate Report**: Runs the report generation script.
8. **Create CML Report**: Publishes the generated report.
9. **Upload Plots**: Uploads validation loss and accuracy plots as artifacts.

## Docker Support

The project includes a `Dockerfile` for containerizing the application, ensuring consistent environments across different systems.

### Building the Docker Image

```bash
docker build -t lightning-template-hydra .
```

### Running the Docker Container

```bash
docker run -it --rm lightning-template-hydra
```

The Docker setup handles dependency installation and sets up the necessary environment variables.

## Testing

Comprehensive testing ensures the reliability and correctness of the codebase. Tests are written using `pytest` and are located in the `tests/` directory.

### Running Tests

```bash
pytest
```

### Test Structure

- **`tests/test_train.py`**: Tests for the training process.
- **`tests/test_datamodule.py`**: Tests for the data module.
- **`tests/test_model.py`**: Tests for the model's forward and training steps.
- **`tests/conftest.py`**: Fixtures for setting up tests.

## License

This project is licensed under the [MIT License](LICENSE).

---
