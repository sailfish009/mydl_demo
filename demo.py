from mydl.arg.parser import get_args
from mydl.config import cfg
from mydl.custom.impl import get_logger
from mydl.modeling.backbone.backbone import build_resnet_backbone

def main():
    # print('hello, world')
    args = get_args("sample")

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = get_logger(cfg, "", 0)
    model = build_resnet_backbone(cfg)


if __name__ == '__main__': main()
