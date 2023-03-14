import os
from domainbed.lib.misc import get_tsne_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSNE plotter')
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    fig = get_tsne_plot(args.csv_path)
    fig.save_fig(os.path.join(args.output_path))