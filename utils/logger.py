# -*- coding: utf-8 -*-
import logging
import os
import sys
import datetime


def init_logger(log_path, name='dispnet'):
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)

    fmt = '%(asctime)s-%(name)s-%(levelname)s-%(message)s'
    formatter = logging.Formatter(fmt)

    logfile = os.path.join(log_path, '%s-%s.log' % (name, datetime.datetime.today()))
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    root.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter(formatter)
    root.addHandler(consoleHandler)

    logging.debug('Logging to %s' % logfile)


class Log(object):
    def __init__(self, log_path, name):
        logfile = os.path.join(log_path, '{}-{}.log'.format(name, datetime.datetime.today()))
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.NOTSET)
        self.fh = logging.FileHandler(logfile)
        self.fh.setLevel(logging.INFO)
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.DEBUG)
        fmt = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s %(message)s')
        self.fh.setFormatter(fmt)
        self.ch.setFormatter(fmt)
        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def __del__(self):
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)
        self.fh.close()
        self.ch.close()

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)


if __name__ == '__main__':
    for i in range(10):
        log = Log(log_path='', name='test_{}'.format(i))
        log.info('info msg1000013333')
        log.debug('debug msg')
        log.warning('warning msg')
        log.error('error msg')
