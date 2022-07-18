import argparse

parser = argparse.ArgumentParser(description='x.py')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--seed', type=int, default=42, help='seed')
parser.add_argument('--Y', type=int, default=6, help='Y - number of events in small window')
parser.add_argument('--validation', type=bool, default=True, help='use validation set')
parser.add_argument('--testing', type=bool, default=False, help='use testing set')
parser.add_argument('--optim', type=str, default='AdamW', help='optimizer to use', choices=['AdamW', 'SGD'])

args = parser.parse_args()