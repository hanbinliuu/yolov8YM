#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import logging.config
import os

from iot_lib.dict_utils import keys_exists


class LoggerConfigurator:
    def __init__(self, fname='log_settings.json', handlerFileName=''):
        self.config_fname = fname
        self.handler_fname = handlerFileName

        # mkdir logs/
        logdir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        # load settings
        try:
            with open(self.config_fname, 'r') as f:
                config = json.load(f)
                if self.handler_fname is not None and len(self.handler_fname) > 0:
                    expected_keys = ['handlers', 'fileHandler', 'filename']
                    if keys_exists(config, *expected_keys):
                        config['handlers']['fileHandler']['filename'] = self.handler_fname
                logging.config.dictConfig(config)
        except FileNotFoundError as fnf_ex:
            print(fnf_ex)

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)
