

import argparse

def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--path_graph", type=str, default='./data/', help="Path of datasets")
    parser.add_argument("--dataset", type=str, default="Texas", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=2, help="Random seed for train-test split. Default is 2.")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")

    parser.add_argument("--early-stop", type=bool, default=False, help="If early stop")
    parser.add_argument("--patience", type=int, default=100, help="Patience for early stop")

    parser.add_argument("--n_repeated", type=int, default=10, help="Number of repeated times. Default is 10.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay") #me
    parser.add_argument("--num_epoch", type=int, default=100, help="Number of training epochs. Default is 100.")
    parser.add_argument("--inner_epochs", type=int, default=1, help="epochs for training")

    parser.add_argument("--data-split-mode", type=str, default="Ratio", help="Data split mode: Number or Ratio")
    parser.add_argument("--train_ratio", type=int, default=0.48, help="Train data ratio. Default is 0.48.")
    parser.add_argument("--valid_ratio", type=int, default=0.32, help="Valid data ratio. Default is 0.32.")

    parser.add_argument("--hdim", nargs='+', type=int, default=512, help="Number of hidden dimensions")
    parser.add_argument('--layers', type=int, default=4, help='Layer number')
    parser.add_argument("--dropout", type=float, default=0.7, help="Dropout rate.")
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off parameter')

    parser.add_argument('--is_energy', type=int, default=1, help='Whether to initiate energy calibration')
    parser.add_argument('--energy_epochs', type=int, default=1, help='Iteration number of energy calibration')
    parser.add_argument('--add_edge', type=int, default=5, help='The number of add edge of each node')
    parser.add_argument('--temperature', type=int, default=0.1, help='Gumbel-Softmax parameter')
    parser.add_argument('--sampling_rate', type=float, default=0.2, help='Sampling ratio')
    args = parser.parse_args()

    return args