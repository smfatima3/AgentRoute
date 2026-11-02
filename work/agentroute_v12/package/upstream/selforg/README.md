# Stochastic Self-Organization in Multi-Agent Systems [ICLR 2026]

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=rS3Jb9AAej)
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-f31f1f)](https://openreview.net/forum?id=rS3Jb9AAej)
[![arXiv](https://img.shields.io/badge/arXiv-2510.00685-teal)](https://arxiv.org/abs/2510.00685)


> [**Stochastic Self-Organization in Multi-Agent Systems [ICLR 2026]**](https://openreview.net/forum?id=rS3Jb9AAej)<br>
> [Nurbek Tastan](https://tnurbek.github.io/), [Samuel Horvath](https://sites.google.com/view/samuelhorvath/home), [Karthik Nandakumar](https://www.cse.msu.edu/~nandakum/)<br>
> The Fourteenth International Conference on Learning Representations (ICLR), 2026<br> 


## Abstract 

Large Language Models (LLMs) have enabled multi-agent systems (MAS) where agents collaborate to solve tasks beyond the reach of a single model. Yet most existing approaches rely on fixed topologies, pretrained graph generators, optimization over edges, or external LLM judges, thereby adding complexity. We introduce a response-conditioned framework that adapts communication on the fly. Agents independently generate answers and assess peer contributions using a Shapley~value-inspired approximation. A directed acyclic graph (DAG) is then constructed to route information from high-contribution agents to others, ensuring stable and efficient message passing without the need for additional supervision or training. We provide a theoretical analysis showing that multiple agents increase the chance of correctness and that the correct answers naturally dominate information flow. Experiments with both strong and weak LLM backends demonstrate robust performance, with significant gains in the weak regime where prior methods collapse.


## Code 

### Structure

```text
selforg/
├── inference.py
├── evaluate.py
├── evaluations/
│   ├── __init__.py
│   └── evaluate_xverify.py
├── methods/
│   ├── __init__.py
│   ├── utils.py
│   ├── mas_base/
│   │   ├── __init__.py
│   │   └── mas_base.py
│   └── selforg/
│       ├── __init__.py
│       ├── selforg_main.py
│       └── configs/
│           └── config_main.yaml
├── utils/
│   ├── __init__.py
│   └── utils.py
├── datasets/
│   ├── download_train_sets.py
│   └── data/
│       └── example_math.json
└── model_api_configs/
    └── model_api_config.json

```

### Model API Config

Use:
- `model_api_configs/model_api_config.json` as template

The alias you pass to `--model_name` must exist as a top-level key in this file.


### Run Inference


```bash
python inference.py \
  --method_name <method> \
  --model_name <model> \
  --model_api_config model_api_configs/model_api_config.json \
  --test_dataset_name <dataset> \
  --output_path <output>
```


### Quick one-sample debug run:

```bash
python inference.py --method_name <method> --debug
```


### Run Evaluation

```bash
python evaluate.py \
  --eval_protocol <protocol> \
  --model_name <model> \
  --model_api_config model_api_configs/model_api_config.json \
  --tested_dataset_name <dataset> \
  --tested_method_name <method> \
  --tested_mas_model_name <model> 
```




## 📖 Citation 
If you like our work, please consider citing us: 

```bibtex
@inproceedings{tastan2026stochastic,
    title={Stochastic Self-Organization in Multi-Agent Systems},
    author={Nurbek Tastan and Samuel Horv{\'a}th and Karthik Nandakumar},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=rS3Jb9AAej}
}
```



## Acknowledgements

We would like to thank the authors of [MASLab](https://github.com/MASWorks/MASLab) for open-sourcing their code. 