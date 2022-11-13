import os
import sys
sys.path.append('core')
import time
import logging
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.utils import ensure_folder


class Logger:
    def __init__(self, args, main="main"):
        self.name = args.name
        if self.name == "":
            self.not_save_board = True
            self.not_save_log = True
        else:
            self.not_save_board = args.not_save_board
            self.not_save_log = args.not_save_log

        self.log_path = args.log_path
        self.debug = args.debug

        self.each_steps = 0
        self.running_loss = {}

        self.last_time = None
        self.writer = None
        self.logger = None
        self.logger_main = main
        self.log_dir = self.log_path
        if not self.not_save_board or not self.not_save_log:
            ensure_folder(self.log_dir)

    def _log_summary(self, index):
        metrics_data = [self.running_loss[k] / self.each_steps for k in self.running_loss.keys()]
        keys = self.running_loss.keys()

        # metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        metrics_str = ""
        for data, key in zip(metrics_data, keys):
            metrics_str += "{}:{:8.6f}, ".format(key, data)
        latest_time = time.time()
        metrics_str += "time:{:8.6f}s.".format(latest_time - self.last_time)
        self.last_time = latest_time

        # print the training status
        self.log_info("Summary {}, {}".format(index, metrics_str), "trainer")

    def _write_summary(self, index):
        if self.not_save_board: 
            return

        if self.writer is None:
            self.init_writer()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.each_steps, index)

    def _clear_summary(self):
        for k in self.running_loss:
            self.running_loss[k] = 0.0
        self.each_steps = 0
        self.last_time = None

    def push(self, metrics, group=None, last=True):
        if last is True:
            self.each_steps += 1

        if self.last_time is None:
            self.last_time = time.time()

        for key in metrics:
            if group is not None:
                loss_key = "{}/{}".format(group, key)
            else:
                loss_key = key

            if loss_key not in self.running_loss:
                self.running_loss[loss_key] = 0.0
            self.running_loss[loss_key] += metrics[key]

    def summary(self, index):
        self._log_summary(index)
        self._write_summary(index)
        self._clear_summary()

    def write_dict(self, index, results, group=None):
        if self.not_save_board:
            return

        if self.writer is None:
            self.init_writer()

        for key in results:
            if group is not None:
                self.writer.add_scalar("{}/{}".format(group, key), results[key], index)
            else:
                self.writer.add_scalar(key, results[key], index)

    def write_image(self, index, name, image):
        if self.not_save_board:
            return

        if self.writer is None:
            self.init_writer()
        grid = torchvision.utils.make_grid(image)
        self.writer.add_image(name, grid, index)

    def init_writer(self):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def init_logger(self, name=None):
        log_path = os.path.join(self.log_dir, "{}.log".format(self.name if name is None else name))

        self.logger = logging.getLogger(self.logger_main)
        self.logger.setLevel(logging.DEBUG)

        if not self.not_save_board: 
            handler = logging.FileHandler(log_path)
            formatter = logging.Formatter('[%(asctime)s-%(name)s-%(levelname)s]: %(message)s', \
                datefmt='%Y/%m/%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%m/%d %H:%M:%S')
        stream_handler.setFormatter(formatter)
        if self.debug:
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

    def log_error(self, error, subname=None):
        if self.logger is None:
            self.init_logger()
        if subname is not None:
            logger = logging.getLogger("{}.{}".format(self.logger_main, subname))
            logger.error(error)
        else:
            self.logger.error(error)

    def log_warn(self, warn, subname=None):
        if self.logger is None:
            self.init_logger()
        if subname is not None:
            logger = logging.getLogger("{}.{}".format(self.logger_main, subname))
            logger.warning(warn)
        else:
            self.logger.warning(warn)

    def log_info(self, info, subname=None):
        if self.logger is None:
            self.init_logger()
        if subname is not None:
            logger = logging.getLogger("{}.{}".format(self.logger_main, subname))
            logger.info(info)
        else:
            self.logger.info(info)

    def log_debug(self, debug, subname=None):
        if self.logger is None:
            self.init_logger()
        if subname is not None:
            logger = logging.getLogger("{}.{}".format(self.logger_main, subname))
            logger.debug(debug)
        else:
            self.logger.debug(debug)

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.logger is not None:
            logging.shutdown()
            self.logger = None

    def is_init_writer(self):
        return self.writer is not None

    def is_init_logger(self):
        return self.logger is not None
    
    def is_init(self):
        return self.is_init_writer() and self.is_init_logger()

    def __del__(self):
        self.close()
