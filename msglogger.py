# -*- coding: utf-8 -*-

import logging
import logging.config
import os

base_dir = os.path.split(os.path.realpath(__file__))[0]
CUR_PATH = base_dir

logger = logging.getLogger("debug")


def init_log(log_path, log_name, log_level="DEBUG"):
    ''''''
    log_level = log_level.upper()

    LOG_PATH_DEBUG = "%s/%s_debug.log" % (log_path, log_name)

    # 日志文件大小
    LOG_FILE_MAX_BYTES = 1 * 512 * 1024 * 1024
    # 备份文件个数
    LOG_FILE_BACKUP_COUNT = 365

    log_conf = {
        "version": 1,
        "formatters": {
            "format1": {
                "format": '%(asctime)-15s [%(thread)d] - [%(filename)s %(lineno)d] %(levelname)s %(message)s',
            },
        },

        "handlers": {

            "handler1": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "format1",
                "maxBytes": LOG_FILE_MAX_BYTES,
                "backupCount": LOG_FILE_BACKUP_COUNT,
                "filename": LOG_PATH_DEBUG
            },

        },

        "loggers": {

            "debug": {
                "handlers": ["handler1"],
                "level": log_level
            },
        },
    }
    logging.config.dictConfig(log_conf)