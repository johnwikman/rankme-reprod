import argparse

class MethodArg:
    def __init__(self, *args, default=None, **kwargs):
        self._args = args
        self._kwargs = kwargs | {"default": default}

    @property
    def default(self): return self._kwargs["default"]


class MethodArguments:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(v, MethodArg)

        self._meth_kwargs = kwargs

    @property
    def defaults(self):
        return type("defaults", (object,), {k: v.default for k, v in self._meth_kwargs.items()})

    def add_to_argparser(self, parser):
        for dest, ma in self._meth_kwargs.items():
            parser.add_argument(*ma._args, dest=dest, **ma._kwargs)
        return parser

    # C++ stream-style operators
    def __rshift__(self, other):
        # margs >> parser
        if isinstance(other, argparse.ArgumentParser):
            return self.add_to_argparser(other)
        else:
            return NotImplemented

    def __rlshift__(self, other):
        # parser << margs
        return self.__rshift__(other)
