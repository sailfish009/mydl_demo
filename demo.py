from mydl.arg.parser import get_args
from mydl.custom.impl import get_device, get_config, get_logger, get_checkpointer
from mydl.custom.impl import mixed_precision, get_arguments, get_data_loder_val
from mydl.data import make_data_loader
from mydl.solver import make_lr_scheduler, make_optimizer
from mydl.modeling.detector import build_detection_model
from mydl.engine.trainer import do_train
from mydl.utils.comm import is_main_process

def main():

    # load config from file and command-line arguments
    args = get_args("sample")
    cfg = get_config(args)

    # logger
    logger = get_logger(cfg, "", 0)

    # model
    model = build_detection_model(cfg)
    device = get_device(cfg)
    model.to(device)

    # optimizer, scheduler
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    model, optimizer = mixed_precision(cfg.DTYPE, model, optimizer)

    # checkpointer
    save_to_disk = is_main_process()
    checkpointer, extra_checkpoint_data\
        = get_checkpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    arguments = get_arguments(extra_checkpoint_data)

    # data loader
    loader = make_data_loader(cfg, is_train=True, is_distributed=False, start_iter= arguments["iteration"])
    loader_val, test_period, checkpoint_period = get_data_loder_val(cfg)

    do_train(cfg, model, loader, loader_val, optimizer, scheduler, checkpointer,
             device, checkpoint_period, test_period, arguments)


if __name__ == '__main__': main()
