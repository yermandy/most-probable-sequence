import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=int, default=0, help="split")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--Y", type=int, default=6, help="Y - number of events in small window"
    )
    parser.add_argument(
        "--validation",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use validation set",
    )
    parser.add_argument(
        "--testing",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use testing set",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="optimizer to use",
        choices=["AdamW", "SGD", "BMRM"],
    )
    parser.add_argument(
        "--biases_only",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="learn only biases",
    )
    parser.add_argument(
        "--cross_validation_fold", type=int, default=-1, help="Use cross validation"
    )
    parser.add_argument("--reg", type=float, default=1, help="regularization for bmrm")
    parser.add_argument(
        "--tol_rel",
        type=float,
        default=0.01,
        help="relative tolerance for bmrm (percents)",
    )
    parser.add_argument(
        "--combine_trn_and_val",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="combine training and validation sets",
    )
    parser.add_argument(
        "--normalize_X",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="normalize samples from X to unit length",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="files/031_more_validation_samples",
        help="root with files",
    )
    parser.add_argument(
        "--outputs_folder", type=str, default=None, help="specific folder for outputs"
    )
    parser.add_argument(
        "--wandb_mode", type=str, default="disabled"
    )

    return parser.parse_args()
