import functools

from torch.utils.data import default_collate


def collate_fn_wrapper(batched_transform):
    @functools.wraps(batched_transform)
    def collate_fn(batch):
        x, y = default_collate(batch)
        x, _ = batched_transform(x, y)
        return (x, y)

    return collate_fn
