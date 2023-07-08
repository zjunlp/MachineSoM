# MachineSoM
Code for the paper [*Machine Society-of-Mind: Exploring Collaboration Mechanisms Among Large Language Models*](https://github.com/zjunlp/MachineSoM).

![settings](figs/settings.jpg)


The Society of Mind (SoM) concept, namely the emergence of intelligence from collaborative and communicative computational modules, enables humans to collaborate and complete complex tasks effectively. In this work, we examine the open question of SoM in modern NLP systems from an empirical and theory-based perspective. We simulate different societies of LLM agents with two **traits** via diverse **collaboration processes and interactive strategies**. On two benchmark datasets, MMLU and Chess, we observe that the LLM agents possess an impressive capacity for collaboration, which can efficiently complete tasks through different social behaviors such as** debate and self-reflection**.


# Requirements
```bash
conda create -n masom python=3.9
pip install -r requirements.txt
```
Datasets: [MMLU](https://huggingface.co/datasets/cais/mmlu), [Chess Validity](https://github.com/google/BIG-bench/blob/761845c22056c885429efd2cfcec345ae00c1de7/bigbench/benchmark_tasks/chess_state_tracking/synthetic_short/task.json), [MATH](https://github.com/hendrycks/math)

# Citation
```bibtex
@misc{MachineSoM2023,
  author = {Jintian Zhang, Xin Xu, Ningyu Zhang, Huajun Chen},
  title = {{Machine Society-of-Mind: Exploring Collaboration Mechanisms Among Large Language Models}},
  month = jul,
  year = {2023},
  url = {https://github.com/zjunlp/MachineSoM}
}
```
