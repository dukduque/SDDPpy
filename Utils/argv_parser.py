'''
https://gist.github.com/Perlence/11284577
'''

"""Parse positional and keyword arguments from `sys.argv`.
We all like :class:`argparser.ArgumentParser`. It is a great tool to build
command-line interfaces with many nice features and automatic coercing.
But sometimes it's just too verbose. At times all we need is to pass arguments
to a function from CLI. And I will help you.
Consider this ``square.py`` program::
    def main(x, exp=2):
        print x ** exp
Now I'd like to pass *x* to :func:`main` as command-line argument. Let's modify
our program a bit::
    from simple_argparse import run_with_args
    def main(x, exp=2):
        print x ** exp
    if __name__ == '__main__':
        run_with_args(main)
Okay, I can no longer hesitate to find out what square of 11 is::
    $ python square.py 11
    121
Majestic! And also we can pass keyword arguments::
    $ python square.py 11 3
    1331
    $ python square.py 11 --exp=3
    1331
Note that keyword arguments without ``={value}`` part are assumed as keywords
with ``True`` value.
By the way, we can also get usage information::
    $ python square.py --help
    usage: square.py [--help] x [exp=2]
Under the covers arguments go through :func:`ast.literal_eval` so they will be
coerced to built-in types.
"""
import sys
from ast import literal_eval


def run_with_args(func, args=None):
    """Parse arguments from ``sys.argv`` or given list of *args* and pass
    them to *func*.
    If ``--help`` is passed to program, print usage information.
    """
    args, kwargs = parse_args(args)
    if kwargs.get('help'):
        from inspect import getargspec
        argspec = getargspec(func)
        if argspec.defaults:
            defaults_count = len(argspec.defaults)
            args = argspec.args[:-defaults_count]
            defaults = zip(argspec.args[-defaults_count:], argspec.defaults)
        else:
            args = argspec.args
            defaults = []
        usage = 'usage: %s [--help]' % sys.argv[0]
        if args:
            usage += ' ' + ' '.join(args)
        if defaults:
            usage += ' ' + ' '.join(('[%s=%r]' % pair for pair in defaults))
        if argspec.varargs:
            usage += ' ' + '*' + argspec.varargs
        if argspec.keywords:
            usage += ' ' + '**' + argspec.keywords
        print(usage)
    else:
        return func(*args, **kwargs)


def parse_args(args=None):
    """Parse positional and keyword arguments from ``sys.argv`` or given list
    of *args*.
    :param args: list of string to parse, defaults to ``sys.argv[1:]``.
    :return: :class:`tuple` of positional args and :class:`dict` of keyword
        arguments.
    Positional arguments have no specific syntax. Keyword arguments must be
    written as ``--{keyword-name}={value}``::
        >>> parse_args(['1', 'hello', 'True', '3.1415926', '--force=True'])
        ((1, 'hello', True, 3.1415926), {'force': True})
    """
    if args is None:
        args = sys.argv[1:]

    positional_args, kwargs = (), {}
    for arg in args:
        if arg.startswith('--'):
            arg = arg[2:]
            try:
                key, raw_value = arg.split('=', 1)
                value = parse_literal(raw_value)
            except ValueError:
                key = arg
                value = True
            kwargs[key.replace('-', '_')] = value
        else:
            positional_args += (parse_literal(arg),)

    return positional_args, kwargs


def parse_literal(string):
    """Parse Python literal or return *string* in case :func:`ast.literal_eval`
    fails."""
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError):
        return string