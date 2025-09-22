# energy-based-am-presentation

Code and figures for my lab presentation on Energy-based Associative memories.
Implementation of Energy-based Associative Memories comes from [Krotov et al. (2025)](https://tutorial.amemory.net/).
Please check out this tutorial and associated notebooks. It is very wonderful.

# Figures

The following figures are generated

## Correlation Matrix Associative Memories

![](./figures/correlation_am/amari_kohonen.png)

## Classical Hopfield Energy

![](./figures/hopfield_am/hopfield_recall_energy.png)

![](./figures/hopfield_am/hopfield_recall_states.png)

## Minerva2

![](./figures/minerva_am/minerva_recall_energy.png)

![](./figures/minerva_am/minerva_recall_states.png)

# Installation and Hacking

Download and install [`uv`](https://docs.astral.sh/uv/)

Enter into the cloned repository, and run the following command:
```sh
$ uv sync
```

To generate the figures, run
```sh
$ uv run main.py
```