# coding=utf-8
# Logs and errors
import time


def raise_value_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nValueError: " + msg + "\n")
    raise ValueError(msg)


def raise_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nError: " + msg + "\n")
    raise Exception(msg)


def raise_import_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nImportError: " + msg + "\n")
    raise ImportError(msg)


def raise_not_implemented_error(msg, logger=None):
    if logger is not None:
        logger.error("\n\nNotImplementedError: " + msg + "\n")
    raise NotImplementedError(msg)


def warning(msg, logger=None):
    if logger is not None:
        logger.warning("\n" + msg + "\n")


def print_toc_message(tic):
    toc = time.time() - tic
    if toc > 60.0:
        if toc > 3600.0:
            toc /= 3600.0
            unit = "hours"
        else:
            toc /= 60.0
            unit = "mins"
    else:
        unit = "sec"
    print("DONE in %f %s!" % (toc, unit))