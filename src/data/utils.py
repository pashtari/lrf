import functools

from torch.utils.data import default_collate


def collate_fn_wrapper(batched_transform):
    @functools.wraps(batched_transform)
    def collate_fn(batch):
        return batched_transform(*default_collate(batch))

    return collate_fn
