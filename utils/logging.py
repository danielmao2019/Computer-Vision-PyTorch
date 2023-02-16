import logging


INDENT = ' ' * 4


def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    # formatter = logging.Formatter(
    #     fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) %(pathname)s F%(funcName)s L%(lineno)s - %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    formatter = logging.Formatter(
        fmt=f"[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(filename=filename)
    handler.setFormatter(formatter)
    handler.setLevel(level=logging.DEBUG)
    logger.addHandler(handler)
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
