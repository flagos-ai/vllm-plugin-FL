# Copyright (c) 2026 BAAI. All rights reserved.
import os
import logging
from logging import Handler, FileHandler
from typing import Dict
import threading

class RankFileHandler(Handler):
    """
    A logging handler that dynamically dispatches log messages to different files
    based on the distributed rank of the process.
    """
    def __init__(self, log_dir: str = "vllm_logs", filename_pattern: str = "rank_{rank}.log", level=logging.NOTSET):
        super().__init__(level)
        # Allow setting custom directory via environment variable VLLM_FL_LOG_DIR
        self.log_dir = os.environ.get("VLLM_FL_LOG_DIR", log_dir)
        self.filename_pattern = filename_pattern
        self.handlers: Dict[int, FileHandler] = {}
        self.lock = threading.RLock()

    def _get_rank(self) -> int:
        # 1. Try torch.distributed
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        # 2. Try common environment variables
        for var in ("RANK", "LOCAL_RANK", "VLLM_FL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
            val = os.environ.get(var)
            if val is not None:
                try:
                    return int(val)
                except ValueError:
                    pass
        return 0

    def emit(self, record):
        try:
            rank = self._get_rank()
            with self.lock:
                if rank not in self.handlers:
                    os.makedirs(self.log_dir, exist_ok=True)
                    filename = self.filename_pattern.format(rank=rank)
                    filepath = os.path.join(self.log_dir, filename)
                    handler = FileHandler(filepath, encoding="utf-8")
                    if self.formatter:
                        handler.setFormatter(self.formatter)
                    self.handlers[rank] = handler
                
                self.handlers[rank].emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        with self.lock:
            for handler in self.handlers.values():
                handler.close()
            self.handlers.clear()
        super().close()


def setup_rank_handlers():
    """
    Add RankFileHandler to 'vllm' and 'vllm_fl' loggers if not already present.
    """
    for logger_name in ("vllm", "vllm_fl"):
        logger = logging.getLogger(logger_name)
        # Check if already patched
        has_rank_handler = any(isinstance(h, RankFileHandler) for h in logger.handlers)
        if not has_rank_handler:
            handler = RankFileHandler()
            
            # Reuse formatting of existing handlers if possible
            formatter = None
            for h in logger.handlers:
                if h.formatter:
                    formatter = h.formatter
                    break
            
            if formatter is None:
                formatter = logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            
            handler.setFormatter(formatter)
            logger.addHandler(handler)


def apply_rank_logging_patch():
    """
    Apply patches to vllm.logger and vllm_fl's logger manager to enable rank-based logging.
    """
    # 1. Patch vllm.logger
    try:
        import vllm.logger
        if not getattr(vllm.logger, "_rank_logging_patched", False):
            _orig_init_logger = vllm.logger.init_logger

            def patched_init_logger(name: str):
                logger = _orig_init_logger(name)
                try:
                    setup_rank_handlers()
                except Exception:
                    pass
                return logger

            vllm.logger.init_logger = patched_init_logger
            vllm.logger._rank_logging_patched = True
    except ImportError:
        pass

    # 2. Patch vllm_fl.dispatch.logger_manager
    try:
        import vllm_fl.dispatch.logger_manager as lm
        if not getattr(lm, "_rank_logging_patched", False):
            _orig_get_logger = lm.get_logger

            def patched_get_logger(name: str = "vllm_fl.dispatch"):
                logger = _orig_get_logger(name)
                try:
                    setup_rank_handlers()
                except Exception:
                    pass
                return logger

            lm.get_logger = patched_get_logger
            lm._rank_logging_patched = True
    except ImportError:
        pass

    # 3. Perform initial setup
    try:
        setup_rank_handlers()
    except Exception:
        pass
