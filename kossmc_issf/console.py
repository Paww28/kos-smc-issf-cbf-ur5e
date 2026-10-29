"""Terminal output: `info` is always printed, `debug` only with --debug (troubleshooting text)."""

_debug = False


def set_debug(on):
    global _debug
    _debug = bool(on)


def is_debug():
    return _debug


def info(*args, **kwargs):
    print(*args, **kwargs)


def debug(*args, **kwargs):
    if _debug:
        print(*args, **kwargs)
