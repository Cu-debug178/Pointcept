import logging

logger_initialized = {}


def get_root_logger(name="pointcept", log_level=logging.INFO, log_file=None, file_mode="w"):
    if name in logger_initialized:
        return logger_initialized[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    
    logger_initialized[name] = logger
    return logger
