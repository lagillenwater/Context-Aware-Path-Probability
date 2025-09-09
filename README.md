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

