from typing import Union, Sequence, Callable
from contextlib import contextmanager
from functools import partial


@contextmanager
def null_context():
    yield


def wrap_class(obj: Union[Sequence, Callable]) -> Callable:
    assert isinstance(
        obj, (Sequence, Callable)
    ), f"{obj} should be a sequence or callable."

    if isinstance(obj, Sequence) and isinstance(obj[0], Callable):
        args = []
        kwargs = {}
        for i, a in enumerate(obj):
            if i == 0:
                callable_obj = a
            elif isinstance(a, Sequence):
                args.extend(a)
            elif isinstance(a, dict):
                kwargs.update(a)

        return partial(callable_obj, *args, **kwargs)
    else:
        return obj
