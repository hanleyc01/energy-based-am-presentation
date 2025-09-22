import typing as T
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import graph_utils as gut

DATA_DIR = "./data/mnist"
FIG_DIR = "./figures"
if not Path(FIG_DIR).exists():
    Path(FIG_DIR).mkdir()

### MODELS #####################################################################

#### CORRELATION MODELS ########################################################


@dataclass
class CorrelationAM:
    """Correlation matrix based AM, like Hopfield.

    Attributes:
        W np.ndarray:
            `(d,d)` weight matrix.
    """

    W: np.ndarray

    def g(self, x: np.ndarray) -> np.ndarray:
        """Partially defined function to be defined by subclasses for the
        function to be applied after the query is sent through the weights.
        """
        ...

    @classmethod
    def from_patterns(cls, Xi: np.ndarray):
        """Initialize a new Correlation-matrix AM based on pattern matrix `Xi`.

        Args:
            Xi np.ndarray:
                `(n,d)` pattern matrix of `n` patterns of `d` dimension.
        """

        W = Xi.T @ Xi
        return cls(W=W)

    def recall(self, x: np.ndarray) -> np.ndarray:
        """Recall the memorized pattern associated with the stored variant
        of `x`, where `x` is a noisy, degraded, masked, or full memorized cue.
        """
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=-1)
        return self.g(self.W @ x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.recall(x)


class Amari(CorrelationAM):
    """Shun-Ichi Amari-John Hopfield network, in the forward-pass one-shot
    formulation. For the energy-based implementation, see `Hopfield.`
    """

    def g(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)


class Kohonen(CorrelationAM):
    """Kohonen Correlation AM with identity function."""

    def g(self, x: np.ndarray) -> np.ndarray:
        return x


#### ENERGY MODELS #############################################################


@dataclass
class EnergyAM:
    Xi: np.ndarray
    n: int

    def energy(self, sigma: np.ndarray) -> float:
        """Return the energy given the current state."""
        ...

    def async_update(self, sigma: np.ndarray, idx: int) -> tuple[np.ndarray, float]:
        """Flip the value `x[idx]` if it lowers the energy."""
        sigma_flipped = np.copy(sigma)
        sigma_flipped[idx] *= -1
        flipped_energy = self.energy(sigma_flipped)
        curr_energy = self.energy(sigma)

        if (flipped_energy - curr_energy) < 0:
            return (sigma_flipped, flipped_energy)
        else:
            return (sigma, curr_energy)

    def async_recall(
        self, sigma0: np.ndarray, nupdates: int = 10_000, seed: int = 10
    ) -> tuple[np.ndarray, list[float]]:
        """Perform asynchronous recall on `sigma0` for `nupdate` steps."""
        rng = np.random.default_rng(seed=seed)

        curr_sigma = sigma0
        energies = []
        idxs = rng.choice(np.arange(sigma0.shape[-1]), size=(nupdates,))

        for idx in idxs:
            curr_sigma, curr_energy = self.async_update(curr_sigma, idx)
            energies.append(curr_energy)

        return curr_sigma, energies


@dataclass
class Hopfield(EnergyAM):
    Xi: np.ndarray
    n: int = 2

    def F_n(self, sims: np.ndarray) -> np.ndarray:
        return sims**self.n

    @T.override
    def energy(self, sigma: np.ndarray) -> float:
        if len(sigma.shape) == 1:
            sigma = np.expand_dims(sigma, axis=-1)
        sims = self.F_n(self.Xi @ sigma)
        sum = np.sum(sims, axis=0)
        return float((-(1 / self.n) * sum)[0])


### GENERATING FIGURES #########################################################


def correlation_am_figures():
    """Generate the Correlation AM figures."""

    fig_path = Path("./figures/correlation_am")
    if not fig_path.exists():
        fig_path.mkdir()

    # Get the training data
    def transform(data):
        data = np.array(data, dtype=np.float64)
        data = rearrange(data, "w h -> (w h)")
        data[data > 0.0] = 1.0
        data[data == 0.0] = -1.0
        return data

    mnist_train = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mnist_data_loader = DataLoader(mnist_train, batch_size=20)
    mnist_it = iter(mnist_data_loader)
    mnist_data, _ = next(mnist_it)

    Xi = mnist_data[:3]
    query = Xi[0]
    masked_query = gut.mask(query, 0.2)

    # Initialize the networks
    amari = Amari.from_patterns(Xi)
    kohonen = Kohonen.from_patterns(Xi)

    # Perform single-pass recall
    amari_recall = amari(masked_query)
    kohonen_recall = kohonen(masked_query)

    # Show the results
    gut.display_recall(
        masked_query,
        amari_recall,
        kohonen_recall,
        ["Masked Query", "Amari Recall", "Kohonen Recall"],
    )
    plt.savefig("./figures/correlation_am/amari_kohonen.png")
    plt.close()


def hopfield_am():
    """Generate Hopfield figures."""
    name = "hopfield_am"
    fig_path = Path(f"./figures/{name}")
    if not fig_path.exists():
        fig_path.mkdir()

    # Get the training data
    def transform(data):
        data = np.array(data, dtype=np.float64)
        data = rearrange(data, "w h -> (w h)")
        data[data > 0.0] = 1.0
        data[data == 0.0] = -1.0
        return data

    mnist_train = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mnist_data_loader = DataLoader(mnist_train, batch_size=20)
    mnist_it = iter(mnist_data_loader)
    mnist_data, _ = next(mnist_it)

    # Initialize the network
    Xi = np.array(mnist_data[:3])
    query = Xi[0]
    masked_query = gut.mask(query, 0.2)
    hopfield = Hopfield(Xi)
    # Perform recall
    final_state, energies = hopfield.async_recall(masked_query)

    # Plot the energies
    plt.plot(energies)
    plt.title("Energy per iteration")
    plt.grid(True, alpha=0.03)
    plt.savefig(f"./figures/{name}/hopfield_recall_energy.png")
    plt.close()

    # Plot the final result
    gut.display_pair(masked_query, final_state, ["Masked Query", "Final State"])
    plt.savefig(f"./figures/{name}/hopfield_recall_states.png")
    plt.close()


def minerva_am():
    """Minerva experiment."""
    name = "minerva_am"
    fig_path = Path(f"./figures/{name}")
    if not fig_path.exists():
        fig_path.mkdir()

    # Get the training data
    def transform(data):
        data = np.array(data, dtype=np.float64)
        data = rearrange(data, "w h -> (w h)")
        data[data > 0.0] = 1.0
        data[data == 0.0] = -1.0
        return data

    mnist_train = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    mnist_data_loader = DataLoader(mnist_train, batch_size=20)
    mnist_it = iter(mnist_data_loader)
    mnist_data, _ = next(mnist_it)

    # Initialize the network
    Xi = np.array(mnist_data[:6])
    query = Xi[0]
    masked_query = gut.mask(query, 0.2)
    minerva = Hopfield(Xi, n=4)  # Note that we are initializing with `n=4`
    # Perform recall
    final_state, energies = minerva.async_recall(masked_query)

    # Plot the energies
    plt.plot(energies)
    plt.title("Energy per iteration")
    plt.grid(True, alpha=0.03)
    plt.savefig(f"./figures/{name}/minerva_recall_energy.png")
    plt.close()

    # Plot the recall
    gut.display_pair(masked_query, final_state, ["Masked Query", "Final State"])
    plt.savefig(f"./figures/{name}/minerva_recall_states.png")
    plt.close()


### MAIN FUNCTION ##############################################################


def main():
    correlation_am_figures()
    hopfield_am()
    minerva_am()


if __name__ == "__main__":
    main()
