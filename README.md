# SciGym: Measuring Scientific Capabilities of Language Models with a Systems Biology Dry Lab

## Setup

To create the conda environment, run the following commands:

```bash
conda create --name scigym python=3.10.16 -y
conda activate scigym

pip install -e .

# Required for graph edit distance metric
conda install --channel conda-forge pygraphviz

# Optional development tools
pip install pre-commit
pre-commit install
```

## Download Benchmark Dataset

We host our full benchmark suite on HuggingFace and provide a script to download it. We provide two splits of the benchmark dataset:

- `small`: Consists of 137 models we evaluated in our paper
- `large`: Consists of an additional 213 models we did not evaluate

To download the splits, run the following commands:

```bash
python data/download.py --split small --save_dir <path_to_save_dir>
python data/download.py --split large --save_dir <path_to_save_dir>
```

## Setup an Agent

You can either use one of our [supported agents](#currently-supported-agents) or set up your own agent. To set up your own agent, you need to implement the [`LLM`](scigym/api/api.py) interface. We provide implementations for Claude, Gemini, and GPT models in [scigym/agent](scigym/agent) folder, along with examples on how to use these models [below](#running-the-benchmark).

### Currently Supported Agents

- `gemini-2.5-pro-preview-03-25`
- `claude-3-5-haiku-20241022`
- `claude-3-7-sonnet-20250219`

## Running the Benchmark

To run the benchmark, you need two things: an agent and a configuration dict that specifies the required parameters for the input, output, and environment components of the run. We provide a default configuration file in `configs/default.yml` which you can modify to suit your needs. The fields in this configuration file are detailed [below](#configuration-fields).

### Example

See also [scigym/examples](scigym/examples) for more examples.

```python
from scigym.agent import Claude
from scigym.main import setup_controller

model_name = "claude-3-5-haiku-20241022"
controller = setup_controller("config/default.yml", model_name)

system_prompt = controller._create_system_prompt()
llm = Claude(model_name=model_name, system_prompt=system_prompt)

controller.run_benchmark(model=llm)
```

### Configuration Fields

- `benchmark_dir`: The directory where the benchmark instance folder is located. You will have these folders after downloading the [benchmark dataset](#download-benchmark-dataset).
- `test_memorize`: Whether to test the agent's ability to memorize the model in a one-shot setting
- `eval_debug_rounds`: Number of rounds to allow the agent to re-submit its hypothesis SBML if there are errors in the previous submission
- `max_iterations`: The maximum number of actions the agent can take in a single episode
- `experiment_actions_path`: Path to system prompt explaining the experimental actions
- `customized_functions_path`: Path to system prompt explaining the routines that the agent can use
- `output_dir`: The directory where the output files will be saved
