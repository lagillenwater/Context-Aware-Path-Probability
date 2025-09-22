# Context-Aware-Path-Probability

This repository is for algorithm development for context aware path search through biomedical knowledge graphs (BKGs). Several of the scripts build on previous work from the Greene Lab and [hetionet project](https://het.io/), including the [connectivity-search-analyses](https://github.com/greenelab/connectivity-search-analyses) repository, [hetio/hetnetpy](https://github.com/hetio/hetnetpy), and [hetio/hetmatpy](https://github.com/hetio/hetmatpy). 

The main goals of this work include:
* approximating an edge probability prior
* assessing the multiplicative assumption of edges in a path

## Setup Instructions

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git

### Clone the Repository

```bash
git clone https://github.com/lagillenwater/Context-Aware-Path-Probability.git
cd Context-Aware-Path-Probability
```

### Create the Environment

1. Navigate to the environments directory:
   ```bash
   cd environments
   ```

2. Create the conda environment from the environment file:
   ```bash
   conda env create -f environment.yml
   ```
   
   Alternatively, you can use the provided script:
   ```bash
   bash create_env.sh
   ```

3. Activate the environment:
   ```bash
   conda activate CAPP
   ```

## Getting Started

### Running the Initial Setup

After setting up the environment, run the necessary scripts to carry out the analyses. You can do this in two ways:

#### Option 1: Interactive Jupyter Notebook 

1. Start JupyterLab:
   ```bash
   jupyter lab
   ```

2. Navigate to the `notebooks/` directory and open 

3. Run Notebooks in order:
    - [`0_create-hetmat.ipynb`](notebooks/0_create-hetmat.ipynb)
    - ['1_generate-permutations.ipynb'](notebooks/1_generate-permutations.ipynb)  


#### Option 2: Shell Script (For batch processing or HPC environments)

1. Navigate to the scripts directory:

2. Run scripts in order:
   ```bash
   bash 0_create_hetmat.sh
   bash 1_create_permutations.sh
   ```
