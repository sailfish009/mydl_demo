from mydl.custom.impl import get_device, setup, run, build_detection_train_loader
from mydl.modeling import build_model
from mydl.custom.protein.loss import make_loss_module
from mydl.custom.protein.data import make_data_loader
from mydl.custom.protein.trainer import do_train
from mydl.solver import build_lr_scheduler, build_optimizer, build_finetune_optimizer
from mydl.checkpoint import DetectionCheckpointer

import os
import logging
logger = logging.getLogger("mydl")


def train(cfg, model, resume=False):
    device = get_device(cfg)
    loss_fn = make_loss_module(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    arguments = {"epoch": 0}
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    train_data_loader, valid_data_loader=build_detection_train_loader(cfg)

    if cfg.SOLVER.FINETUNE == "on" and arguments["epoch"] == 0:
        finetune_optimizer = build_finetune_optimizer(cfg, model)
        do_train(
            model=model,
            loss_module=loss_fn,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            optimizer=finetune_optimizer,
            scheduler=scheduler,
            checkpointer=checkpointer,
            device=device,
            train_epoch=cfg.SOLVER.TRAIN_EPOCH,
            checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
            is_mixup=cfg.SOLVER.MIXUP,
            arguments=arguments
        )

    do_train(
        model=model,
        loss_module=loss_fn,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpointer=checkpointer,
        device=device,
        train_epoch=cfg.SOLVER.TRAIN_EPOCH,
        checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
        is_mixup=cfg.SOLVER.MIXUP,
        arguments=arguments
    )



def main(args):

    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    train(cfg, model)

    # return do_test(cfg, model)


if __name__ == '__main__': run(main)
