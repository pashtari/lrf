def compress_dct_patches(image, compression_ratio, patch_size=8):
    """Compress the image using DCT on patches."""
    image = torch.tensor(image, dtype=torch.float32).permute(-1, 0, 1)

    *_, H, W = image.shape

    # Rearrange image to form 8x8 patches
    patches = rearrange(
        image, "c (h p1) (w p2) -> c (h w) p1 p2", p1=patch_size, p2=patch_size
    )

    # Apply DCT to each patch
    dct_patches = dct.dct_2d(patches)

    # Keep only a certain number of low-frequency components
    R = math.floor(math.sqrt(patch_size * patch_size / compression_ratio))
    mask = torch.zeros_like(dct_patches)
    mask[:, :, :R, :R] = 1  # Keep top-left corner according to R
    compressed_dct_patches = dct_patches * mask

    # Apply IDCT to reconstruct the patches
    reconstructed_patches = dct.idct_2d(compressed_dct_patches)

    # Rearrange compressed patches back to image format
    reconstructed_image = rearrange(
        reconstructed_patches,
        "c (h w) p1 p2 -> c (h p1) (w p2)",
        h=H // patch_size,
        w=W // patch_size,
    )
    return (
        minmax_normalize(reconstructed_image, 0, 1)
        .permute(1, 2, 0)
        .to(torch.float64)
        .numpy()
    )
