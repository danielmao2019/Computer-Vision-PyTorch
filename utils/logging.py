import logging
import sys


INDENT = ' ' * 4


def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    # A more complete version of formatter
    # formatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    f_handler = logging.FileHandler(filename=filename)
    f_handler.setFormatter(formatter)
    f_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(f_handler)
    s_handler = logging.StreamHandler(stream=sys.stdout)
    s_handler.setFormatter(formatter)
    s_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(s_handler)
    return logger


def log_criterion_info(logger, criterion):
    logger.info(f"criterion={criterion.__class__.__name__}")
    string = criterion.__str__().split('\n')
    for s in string:
        logger.info(INDENT + s)


def log_optimizer_info(logger, optimizer):
    logger.info(f"optimizer={optimizer.__class__.__name__}")
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    for key in sorted(group):
        if key != 'params':
            logger.info(INDENT + f"{key}={group[key]}")
