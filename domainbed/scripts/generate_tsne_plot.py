import sys
sys.path.append('.')

import os
import argparse
from domainbed.lib.misc import get_tsne_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSNE plotter')
    parser.add_argument('--pickle_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    fig = get_tsne_plot(args.pickle_path)
    fig.savefig(os.path.join(args.output_path))