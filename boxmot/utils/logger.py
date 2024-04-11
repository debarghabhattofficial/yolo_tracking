import logging


class Logger:
    @staticmethod
    def get_logger(name, level):
        time_format = "%m/%d/%Y-%H:%M:%S"
        formatter = logging.Formatter(
            fmt="[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s", 
            datefmt=time_format 
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.setLevel(level)  # Set logging level.
        logger.propagate = False

        return logger