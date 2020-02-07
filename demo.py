from mydl.custom.impl import setup, run
from mydl.modeling import build_model
from mydl.custom.protein.loss import make_loss_module
from mydl.custom.protein.data import make_data_loader
# from mydl.solver import build_lr_scheduler, build_optimizer
# from mydl.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

import os
import logging
logger = logging.getLogger("mydl")


def do_train(cfg, model, resume=False):
    loss_fn = make_loss_module(cfg)


def main(args):

    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    do_train(cfg, model)


    # do_train(cfg, model)
    # return do_test(cfg, model)


if __name__ == '__main__': run(main)
