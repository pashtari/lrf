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

from lrf.utils.metrics import psnr, ssim, bits_per_pixel, compression_ratio


def zscore_normalize(
    tensor: torch.Tensor, dim: tuple[int, int] = (-2, -1), eps: float = 1e-8
) -> torch.Tensor:
    """
    Applies Z-score normalization to a tensor along specified dimensions.

    Args:
        tensor (torch.Tensor): Input tensor to normalize.
        dim (tuple[int, int], optional): Dimensions along which to compute mean and std.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Z-score normalized tensor.
    """
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + eps)
    return normalized_tensor


def minmax_normalize(
    tensor: torch.Tensor, dim: tuple[int, int] = (-2, -1), eps: float = 1e-8
) -> torch.Tensor:
    """
    Applies Min-Max normalization to a tensor along specified dimensions.

    Args:
        tensor (torch.Tensor): Input tensor to normalize.
        dim (tuple[int, int], optional): Dimensions along which to compute min and max.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Min-Max normalized tensor.
    """
    min_val = torch.amin(tensor, dim=dim, keepdim=True)
    max_val = torch.amax(tensor, dim=dim, keepdim=True)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + eps)
    return normalized_tensor


def eval_compression(
    image: str | np.ndarray | torch.Tensor,
    encoder: Callable,
    decoder: Callable,
    reconstruct: bool = False,
    **kwargs,
) -> dict:
    """Calculates compression metric values for an image.

    Args:
        image (str, np.ndarray, or torch.Tensor): The image to be evaluated. Can be a file path, a numpy array, or a torch tensor.
        encoder (Callable): The function to encode the image.
        decoder (Callable): The function to decode the encoded image.
        reconstruct (bool, optional): Whether to include the reconstructed image in the output. Defaults to False.
        **kwargs: Additional arguments to pass to the encoder.

    Returns:
        dict: A dictionary containing the compression metrics.
    """

    # Read and preprocess the image
    if isinstance(image, str):
        image = read_image(image)
    elif isinstance(image, np.ndarray):
        image = torch.tensor(
            image.transpose((2, 0, 1))
        )  # Ensure correct shape for PyTorch
    elif not isinstance(image, torch.Tensor):
        raise ValueError("Image must be a file path, numpy array, or torch tensor.")

    # Encode the image and measure time
    profiler = Profiler()
    profiler.start()
    encoded = encoder(image, **kwargs)
    profiler.stop()
    encoding_time = 1000 * profiler.last_session.duration

    # Decode the image and measure time
    profiler.start()
    reconstructed = decoder(encoded)
    profiler.stop()
    decoding_time = 1000 * profiler.last_session.duration

    # Calculate compression metrics
    cr_value = compression_ratio(image, encoded)
    bpp_value = bits_per_pixel(image.shape[-2:], encoded)
    psnr_value = psnr(image, reconstructed).item()
    ssim_value = ssim(image, reconstructed).item()

    # Prepare the output dictionary
    output = {
        "compression ratio": cr_value,
        "bit rate (bpp)": bpp_value,
        "PSNR (dB)": psnr_value,
        "SSIM": ssim_value,
        "encoding time (ms)": encoding_time,
        "decoding time (ms)": decoding_time,
    }

    if reconstruct:
        output["reconstructed"] = reconstructed

    return output


def read_image(*args, **kwargs) -> torch.Tensor:
    """
    Reads an image from the specified path and converts it to a PyTorch tensor.
    Args and kwargs are passed to the imread function.

    Returns:
        torch.Tensor: Image tensor with dimensions rearranged to [C, H, W].
    """
    image = imread(*args, **kwargs)
    image_tensor = torch.tensor(image.transpose((2, 0, 1)))
    return image_tensor


def vis_image(
    image: torch.Tensor,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    prefix: str = "",
    format: str = "pdf",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Visualizes an image tensor and optionally saves it to a file.

    Args:
        image (torch.Tensor): Image tensor with dimensions [C, H, W].
        title (str or None, optional): Title of the image (default: None).
        save_dir (str or None, optional): Directory to save the image (default: None).
        prefix (str, optional): Prefix for the saved image file name (default: "").
        format (str, optional): File format for saving the image (default: "pdf").
        **kwargs: Additional keyword arguments for imshow.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects of the plot.
    """
    if image.ndimension() != 3 or image.shape[0] not in [1, 3]:
        raise ValueError("Image tensor should have shape [C, H, W] with C being 1 or 3.")

    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0), **kwargs)
    ax.axis("off")

    if title:
        ax.set_title(title)

    plt.show()

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{prefix}.{format.lower()}")
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    return fig, ax


def vis_image_batch(
    images: torch.Tensor,
    multi_channels: bool = True,
    grid_size: Optional[int | tuple[int, int]] = None,
    fig_size: Optional[tuple[int, int]] = None,
    title: Optional[str] = None,
    save_dir: Optional[str] = None,
    prefix: str = "",
    format: str = "pdf",
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Visualize a batch of 2D images and optionally saves it to a file..

    Args:
    - images (torch.Tensor): A tensor of shape (*batch_dims, [channel], height, width).
    - multi_channels (bool, optional): Boolean indicating if the images have multiple channels (default: True).
    - grid_size (int, tuple[int, int], or None, optional): The size of the grid for displaying images (default: None).
    - fig_size (tuple[int, int] or None, optional): Size of the figure (default: None).
    - title (str or None, optional): Title for the plot (default: None).
    - save_dir (str or None, optional): Directory to save the figure (default: None).
    - prefix (str, optional): Prefix for the saved file name (default: "").
    - format (str, optional): Format to save the figure (default: "pdf").
    - **kwargs: Additional arguments passed to plt.imshow()

    Returns:
    - tuple[plt.Figure, np.ndarray]: The matplotlib figure object and array of subplot axes
    """
    pass
    shape = images.shape[-2:]

    # Add batch dimension if necessary
    if images.ndim == 2:
        images = images[None, ...]

    # Reorder dimensions if multi-channel
    if multi_channels:
        images = np.einsum("...c mn -> ... mn c", images)
        batch_dims = images.shape[:-3]
    else:
        batch_dims = images.shape[:-2]

    total_subplots = np.prod(batch_dims)

    if grid_size is None:
        num_cols = int(np.ceil(np.sqrt(total_subplots)))
        grid_size = (int(np.ceil(total_subplots / num_cols)), num_cols)

    if fig_size is None:
        fig_height = grid_size[0] * shape[0]
        fig_width = grid_size[1] * shape[1]
        fig_size = (
            10 * fig_width / (fig_height + fig_width),
            10 * fig_height / (fig_height + fig_width),
        )

    # Create subplots
    fig, axs = plt.subplots(*grid_size, figsize=fig_size)

    # Flatten axes for easy indexing
    axs = axs.ravel() if grid_size[0] * grid_size[1] > 1 else [axs]

    # Visualize each image
    for idx in np.ndindex(batch_dims):
        ax_idx = idx[0] if len(batch_dims) == 1 else np.ravel_multi_index(idx, batch_dims)
        ax = axs[ax_idx]
        ax.imshow(images[idx], **kwargs)
        ax.axis("off")

    # Turn off remaining unused subplots
    for ax in axs[total_subplots:]:
        ax.axis("off")

    # Add title if provided
    if title:
        fig.suptitle(title)

    fig.subplots_adjust(
        wspace=0.2 * shape[0] / (shape[0] + shape[1]),
        hspace=0.2 * shape[1] / (shape[0] + shape[1]),
    )
    plt.show()

    # Save figure if save directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{prefix}.{format}").lower(),
            bbox_inches="tight",
            pad_inches=0,
        )

    return fig, axs


class LOESS:
    def __init__(
        self, frac: float | Sequence[float] = 0.3, degree: int | Sequence[int] = 1
    ) -> None:
        """
        Initialize the LOESS model.

        Args:
            frac (float | Sequence[float], optional): Fraction of data points to consider for local regression (default: 0.3).
            degree (int | Sequence[int], optional): Degree of the polynomial to fit (default: 1).
        """
        self.frac = np.atleast_1d(frac)
        self.degree = np.atleast_1d(degree)
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.best_frac: Optional[float] = None
        self.best_degree: Optional[int] = None

    def _tricube(self, d: np.ndarray) -> np.ndarray:
        """
        Tricube weight function.

        Args:
            d (np.ndarray): Distances normalized to [0, 1].

        Returns:
            np.ndarray: Tricube weights.
        """
        return np.clip((1 - d**3) ** 3, 0, 1)

    def _design_matrix(self, x: np.ndarray, degree: int) -> np.ndarray:
        """
        Generate a design matrix for polynomial regression.

        Args:
            x (np.ndarray): Input values.
            degree (int): Degree of the polynomial.

        Returns:
            np.ndarray: Design matrix.
        """
        return np.vander(x, degree + 1)

    def _loocv(self, frac: float, degree: int) -> float:
        """
        Perform leave-one-out cross-validation for given frac and degree.

        Args:
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

    def _find_best_params(self) -> tuple[float, int]:
        """
        Perform grid search to find the best frac and degree.

        Returns:
            tuple[float, int]: Best frac and degree based on LOOCV.
        """
        best_score = float("inf")
        best_params = (None, None)

        for frac, degree in product(self.frac, self.degree):
            score = self._loocv(frac, degree)
            if score < best_score:
                best_score = score
                best_params = (frac, degree)

        return best_params

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LOESS":
        """
        Fit the LOESS model to the data and perform grid search if necessary.

        Args:
            x (np.ndarray): Input values.
            y (np.ndarray): Output values.

        Returns:
            LOESS: Fitted model with optimal frac and degree.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if len(self.frac) > 1 or len(self.degree) > 1:
            self.best_frac, self.best_degree = self._find_best_params()
        else:
            self.best_frac = self.frac[0]
            self.best_degree = self.degree[0]

        return self

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict new values using the fitted LOESS model.

        Args:
            x_new (np.ndarray): New input values.

        Returns:
            np.ndarray: Predicted values.
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
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the Plot object with data.

        Args:
            data (pd.DataFrame, dict[str, Sequence], or Sequence[dict]): The input data.
            columns (Sequence[str] or None, optional): Column names for the data if it is not a DataFrame (default: None).
        """
        self.data = pd.DataFrame(data, columns=columns)
        self.x: Optional[str] = None
        self.y: Optional[str] = None
        self.x_values: Optional[Sequence] = None
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def interpolate(
        self,
        x: str,
        y: str,
        x_values: Sequence,
        groupby: str | Sequence[str] = ("data", "method"),
    ) -> pd.DataFrame:
        """
        Interpolate the data using LOESS.

        Args:
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            x_values (Sequence): The x-values for interpolation.
            groupby (str or Sequence[str], optional): Column(s) to group by (default: ("data", "method")).

        Returns:
            pd.DataFrame: The interpolated data.
        """
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
        xlim: tuple[Optional[float], Optional[float]] = (None, None),
        ylim: tuple[Optional[float], Optional[float]] = (None, None),
        legend_labels: Optional[Sequence[str]] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the interpolated data.

        Args:
            x (str): The column name for the x-axis.
            y (str): The column name for the y-axis.
            groupby (str, optional): The column name for grouping data (default: "method").
            errorbar (str or None, optional): The error bar type (default: "se").
            dashed (bool, optional): Whether to use dashed lines for extrapolated data (default: True).
            xlim (tuple[float | None, float | None], optional): The x-axis limits (default: (None, None)).
            ylim (tuple[float | None, float | None], optional): The y-axis limits (default: (None, None)).
            legend_labels (Sequence[str] or None, optional): The labels for the legend (default: None).

        Returns:
            tuple[plt.Figure, plt.Axes]: The figure and axes of the plot.
        """
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
        return fig, ax

    def save(self, save_dir: str = ".", prefix: str = "", format: str = "pdf") -> None:
        """
        Save the plot to a file.

        Args:
            save_dir (str, optional): The directory to save the plot (default: ".").
            prefix (str, optional): The prefix for the saved file name (default: "").
            format (str, optional): The format to save the file in (default: "pdf").
        """
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
) -> tuple[plt.Figure, np.ndarray]:
    """
    Visualize a collage of reconstructed images for various methods and bit rates.

    Args:
        data (dict[str, Sequence]): A dictionary containing the data to visualize. Expected keys are:
            'data', 'method', 'compression ratio', 'bit rate (bpp)', 'PSNR (dB)', 'SSIM', 'encoding time (ms)', 'decoding time (ms)', 'reconstructed'.
        bpps (Sequence[float]): A sequence of bit rates (bpp) to visualize.
        save_dir (str or None, optional): Directory to save the figure and individual images (default: None).
        prefix (str, optional): Prefix for the saved file name (default: "").
        format (str, optional): Format to save the figure (default: "pdf").

    Returns:
        tuple[plt.Figure, np.ndarray]: The matplotlib figure object and array of subplot axes.
    """

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

    for i, bpp in enumerate(bpps):
        for j, method in enumerate(methods):
            method_idx = np.where(method_array == method)[0]
            min_idx = np.argmin(np.abs(bpp_array[method_idx] - bpp))
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
            if save_dir is not None:
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

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the subplots collage
        fig.savefig(
            os.path.join(save_dir, f"{prefix}_qualitative_comparison.{format}").lower(),
            bbox_inches="tight",
            pad_inches=0,
        )

    return fig, axs


def json_serializer(obj: Any) -> Any:
    """
    Serializes an object to JSON-compatible format.

    Args:
        obj (Any): The object to serialize.

    Returns:
        Any: The serialized object.

    Raises:
        TypeError: If the object is not JSON serializable.
    """
    if isinstance(obj, torch.dtype):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def json_deserializer(dct: dict[str, Any]) -> dict[str, Any]:
    """
    Deserializes a dictionary containing strings that represent torch data types.

    Args:
        dct (dict): The dictionary to deserialize.

    Returns:
        dict: The deserialized dictionary.
    """
    for key, value in dct.items():
        if isinstance(value, str) and value.startswith("torch."):
            dct[key] = getattr(torch, value.split(".")[1])
    return dct


def read_config(file_name: str) -> dict[str, Any]:
    """
    Reads a JSON configuration file and deserializes it.

    Args:
        file_name (str): The name of the configuration file to read.

    Returns:
        dict: The deserialized configuration data.
    """
    with open(file_name, "r") as json_file:
        data = json.load(json_file, object_hook=json_deserializer)

    return data


def save_config(
    data: Sequence[dict[str, Any]],
    save_dir: Optional[str] = None,
    prefix: str = "",
) -> None:
    """
    Saves a sequence of dictionaries to a JSON file.

    Args:
        data (Sequence[dict]): The data to save.
        save_dir (str or None, optional): The directory to save the file in (default: None).
        prefix (str, optional): The prefix for the saved file name (default: "").

    Raises:
        FileNotFoundError: If the save directory does not exist.
    """
    if save_dir is None:
        raise FileNotFoundError("Save directory must be specified")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, f"{prefix}_results.json"), "w") as json_file:
        json.dump(data, json_file, indent=4, default=json_serializer)
