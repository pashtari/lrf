from typing import Any, Callable, Optional, Sequence
import os
import json
import re
from itertools import product


import numpy as np
import pandas as pd
import torch
from scipy.linalg import lstsq
from skimage.io import imread
from pyinstrument import Profiler
import seaborn as sns
import matplotlib.pyplot as plt

from ..compression import get_bbp, get_compression_ratio
from .metrics import psnr, ssim


def eval_compression(
    image: str | np.ndarray | torch.Tensor,
    encoder: Callable,
    decoder: Callable,
    reconstruct: bool = False,
    **kwargs,
) -> dict:
    """Calculates metric values for an image."""

    if isinstance(image, str):
        image = read_image(image)
    elif isinstance(image, np.ndarray):
        image = torch.tensor(image.transpose([-1, 0, 1]))
    elif isinstance(image, torch.Tensor):
        pass
    else:
        raise ValueError

    profiler = Profiler()
    profiler.start()
    enocoded = encoder(image, **kwargs)
    profiler.stop()
    encoding_time = 1000 * profiler.last_session.duration

    profiler = Profiler()
    profiler.start()
    reconstructed = decoder(enocoded)
    profiler.stop()
    decoding_time = 1000 * profiler.last_session.duration

    compression_ratio = get_compression_ratio(image, enocoded)
    bpp = get_bbp(image.shape[-2:], enocoded)
    psnr_value = psnr(image, reconstructed).item()
    ssim_value = ssim(image, reconstructed).item()

    output = {
        "compression ratio": compression_ratio,
        "bit rate (bpp)": bpp,
        "PSNR (dB)": psnr_value,
        "SSIM": ssim_value,
        "encoding time (ms)": encoding_time,
        "decoding time (ms)": decoding_time,
    }

    if reconstruct:
        output["reconstructed"] = reconstructed

    return output


def read_image(*args, **kwargs) -> torch.Tensor:
    image = imread(*args, **kwargs)
    image = torch.tensor(image.transpose([-1, 0, 1]))
    return image


def vis_image(
    image: torch.Tensor,
    save_dir: Optional[str] = None,
    prefix: str = "",
    format: str = "pdf",
) -> tuple[plt.Figure, plt.Axes]:

    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))
    ax.axis("off")
    plt.show()

    if isinstance(save_dir, str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(
            os.path.join(save_dir, f"{prefix}.{format}").lower(),
            bbox_inches="tight",
            pad_inches=0,
        )
    return fig, ax


class LOESS:
    def __init__(self, frac=0.3, degree=1):
        """
        Initialize the LOESS model.

        Parameters:
        frac (float or sequence of floats): Fraction of data points to consider for local regression.
        degree (int or sequence of ints): Degree of the polynomial to fit.
        """
        self.frac = np.atleast_1d(frac)
        self.degree = np.atleast_1d(degree)
        self.x = None
        self.y = None
        self.best_frac = None
        self.best_degree = None

    def _tricube(self, d):
        """
        Tricube weight function.

        Parameters:
        d (array-like): Distances normalized to [0, 1].

        Returns:
        array-like: Tricube weights.
        """
        return np.clip((1 - d**3) ** 3, 0, 1)

    def _design_matrix(self, x, degree):
        """
        Generate a design matrix for polynomial regression.

        Parameters:
        x (array-like): Input values.
        degree (int): Degree of the polynomial.

        Returns:
        array-like: Design matrix.
        """
        return np.vander(x, degree + 1)

    def _loocv(self, frac, degree):
        """
        Perform leave-one-out cross-validation for given frac and degree.

        Parameters:
        frac (float): Fraction of data points to consider for local regression.
        degree (int): Degree of the polynomial.

        Returns:
        float: Mean squared error for LOOCV.
        """
        n = len(self.x)
        errors = np.zeros(n)

        for i in range(n):
            x_train = np.delete(self.x, i)
            y_train = np.delete(self.y, i)
            x_val = self.x[i]
            y_val = self.y[i]

            model = LOESS(frac=frac, degree=degree)
            model.fit(x_train, y_train)
            y_pred = model.predict([x_val])[0]
            errors[i] = (y_val - y_pred) ** 2

        return np.mean(errors)

    def _find_best_params(self):
        """
        Perform grid search to find the best frac and degree.

        Returns:
        tuple: Best frac and degree based on LOOCV.
        """
        best_score = float("inf")
        best_params = (None, None)

        for frac, degree in product(self.frac, self.degree):
            score = self._loocv(frac, degree)
            if score < best_score:
                best_score = score
                best_params = (frac, degree)

        return best_params

    def fit(self, x, y):
        """
        Fit the LOESS model to the data and perform grid search if necessary.

        Parameters:
        x (array-like): Input values.
        y (array-like): Output values.

        Returns:
        self: Fitted model with optimal frac and degree.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if len(self.frac) > 1 or len(self.degree) > 1:
            self.best_frac, self.best_degree = self._find_best_params()
        else:
            self.best_frac = self.frac[0]
            self.best_degree = self.degree[0]

        return self

    def predict(self, x_new):
        """
        Predict new values using the fitted LOESS model.

        Parameters:
        x_new (array-like): New input values.

        Returns:
        array-like: Predicted values.
        """
        x_new = np.asarray(x_new)
        n = len(self.x)
        k = int(np.ceil(self.best_frac * n))

        y_new = np.zeros_like(x_new)

        for i, x in enumerate(x_new):
            distances = np.abs(self.x - x)
            idx = np.argsort(distances)[:k]
            weights = self._tricube(distances[idx] / distances[idx][-1])
            W = np.diag(weights)
            A = self._design_matrix(self.x[idx], self.best_degree)
            B = self.y[idx]
            beta = lstsq(W @ A, W @ B, cond=None)[0]
            y_new[i] = np.polyval(beta, x)  # np.polyval expects highest degree first

        return y_new


class Plot:
    def __init__(
        self,
        data: pd.DataFrame | dict[str, Sequence] | Sequence[dict],
        columns: Optional[Sequence] = None,
    ) -> None:
        self.data = pd.DataFrame(data, columns=columns)
        self.x = None
        self.y = None
        self.x_values = None
        self.fig = None
        self.ax = None

    def interpolate(
        self,
        x: str,
        y: str,
        x_values: Sequence,
        groupby: str | Sequence[str] = ("data", "method"),
    ) -> pd.DataFrame:

        self.x = x
        self.y = y
        self.x_values = x_values
        groupby = [groupby] if isinstance(groupby, str) else groupby

        interp_data = []
        for c, grp in self.data.groupby(groupby):
            grp = grp.drop_duplicates(self.x)
            interp_grp = pd.DataFrame({**dict(zip(groupby, c)), self.x: self.x_values})
            loess = LOESS(frac=np.arange(0.15, 0.75, 0.1), degree=[1, 2])
            loess.fit(grp[self.x], grp[self.y])
            interp_grp[self.y] = loess.predict(self.x_values)

            x_min, x_max = min(grp[self.x]), max(grp[self.x])
            interp_grp["extrapolated"] = (self.x_values < x_min) | (self.x_values > x_max)

            interp_data.append(interp_grp)

        self.data = pd.concat(interp_data)
        return self.data

    def plot(
        self,
        x: str,
        y: str,
        groupby: str = "method",
        errorbar: Optional[str] = "se",
        dashed: bool = True,
        xlim: tuple[float | None, float | None] = (None, None),
        ylim: tuple[float | None, float | None] = (None, None),
        legend_labels: Optional[Sequence] = None,
    ) -> tuple[plt.Figure, plt.Axes]:

        self.x = x
        self.y = y
        legend_labels = (
            tuple(self.data[groupby].unique())
            if legend_labels is None
            else tuple(legend_labels)
        )

        if dashed and "extrapolated" in self.data.columns:
            self.data = pd.concat(
                [
                    grp.assign(dashed=grp["extrapolated"].all())
                    for _, grp in self.data.groupby([groupby, self.x])
                ]
            )
        else:
            self.data["dashed"] = False

        sns.set_theme(style="white")
        fig, ax = plt.subplots()
        sns.lineplot(
            ax=ax,
            data=self.data[~self.data["dashed"]],
            x=self.x,
            y=self.y,
            hue=groupby,
            errorbar=errorbar,
            linestyle="-",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            legend="brief",
        )  # solid line for non-extrapolated region
        sns.lineplot(
            ax=ax,
            data=self.data,
            x=self.x,
            y=self.y,
            hue=groupby,
            errorbar=None,
            linestyle="--",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            legend=False,
        )  # dashed line for extrapolated region
        ax.grid()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        handles, labels = ax.get_legend_handles_labels()
        new_handles, new_labels = zip(
            *((handles[labels.index(label)], label) for label in legend_labels)
        )
        sns.move_legend(ax, "lower right", handles=new_handles, labels=new_labels)

        self.fig, self.ax = fig, ax
        return self

    def save(self, save_dir: str = ".", prefix: str = "", format: str = "pdf"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        metric_name = re.sub(r"\s*\(.*?\)\s*", "", self.y).replace(" ", "_")
        self.fig.savefig(
            os.path.join(save_dir, f"{prefix}_{metric_name}.{format}".lower()),
            bbox_inches="tight",
            pad_inches=0,
        )


def vis_collage(
    data: dict[str, Sequence],
    bpps: Sequence[float],
    save_dir: Optional[str] = None,
    prefix: str = "",
    format: str = "pdf",
) -> tuple[plt.Figure, plt.Axes]:

    data = pd.DataFrame(
        data,
        columns=(
            "data",
            "method",
            "compression ratio",
            "bit rate (bpp)",
            "PSNR (dB)",
            "SSIM",
            "encoding time (ms)",
            "decoding time (ms)",
            "reconstructed",
        ),
    )

    methods = list(dict.fromkeys(data["method"]))
    method_array = np.array(data["method"])
    bpp_array = np.array(data["bit rate (bpp)"])
    psnr_array = np.array(data["PSNR (dB)"])

    # Plot the compressed images for each method and bpp
    fig, axs = plt.subplots(
        len(bpps),
        len(methods),
        figsize=(6 * len(methods), 5 * len(bpps)),
    )

    # Set titles for columns
    for ax, method in zip(axs[0], methods):
        ax.set_title(method, fontsize=24)

    for i, bbp in enumerate(bpps):
        for j, method in enumerate(methods):
            method_idx = np.where(method_array == method)[0]
            min_idx = np.argmin(np.abs(bpp_array[method_idx] - bbp))
            reconstructed_image = [data["reconstructed"][k] for k in method_idx][min_idx]
            axs[i, j].imshow(reconstructed_image.permute(1, 2, 0))
            bpp_ij = bpp_array[method_idx][min_idx]
            psnr_ij = psnr_array[method_idx][min_idx]
            axs[i, j].set_xlabel(
                f"bpp = {bpp_ij:.2f}, PSNR = {psnr_ij:.2f}dB", fontsize=16
            )
            axs[i, j].tick_params(
                axis="both",
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

            # Save each subplot as an individual image
            individual_fig, individual_ax = plt.subplots()
            individual_ax.imshow(reconstructed_image.permute(1, 2, 0))
            individual_ax.axis("off")
            if isinstance(save_dir, str):
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                individual_fig.savefig(
                    os.path.join(
                        save_dir,
                        f"{prefix}_{method}_bpp_{bpp_ij:.2f}_psnr_{psnr_ij:.2f}.{format}",
                    ).lower(),
                    bbox_inches="tight",
                    pad_inches=0,
                )
            plt.close(individual_fig)

    if isinstance(save_dir, str):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the subplots collage
        fig.savefig(
            os.path.join(save_dir, f"{prefix}_qualitative_comparison.{format}").lower(),
            bbox_inches="tight",
            pad_inches=0,
        )

    return fig, axs


def json_serializer(obj) -> Any:
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_deserializer(dct) -> dict:
    for key, value in dct.items():
        if isinstance(value, str) and value.startswith("torch."):
            dct[key] = getattr(torch, value.split(".")[1])
    return dct


def read_config(file_name: str) -> dict:
    with open(file_name, "r") as json_file:
        data = json.load(json_file, object_hook=json_deserializer)

    return data


def save_config(
    data: Sequence[dict],
    save_dir: Optional[str] = None,
    prefix: str = "",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, f"{prefix}_results.json"), "w") as json_file:
        json.dump(data, json_file, indent=4, default=json_serializer)
