import sys
import argparse

from tools.datasets.div2k import div2k

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--scale', help='Specifies the downsampling scale.')
parser.add_argument('--subset', help='Specifies the subset.')
parser.add_argument(
    '--sampling', help='Specifies the downsampling method.')
parser.add_argument(
    '--resolution', help='Specifies the resolution.')


args = parser.parse_args()


if args.scale is None:
    args.scale = div2k.Scale.X2.value

if args.subset is None:
    args.subset = div2k.Subset.TRAINING.value

if args.sampling is None:
    args.sampling = div2k.Sampling.BICUBIC.value

if args.resolution is None:
    args.resolution = div2k.Resolution.HIGH.value


arg_scale = args.scale.upper()
arg_subset = args.subset.upper()
arg_sampling = args.sampling.upper()
arg_resolution = args.resolution.upper()


try:
    scale = div2k.Scale[arg_scale]
    subset = div2k.Subset[arg_subset]
    sampling = div2k.Sampling[arg_sampling]
    resolution = div2k.Resolution[arg_resolution]
except KeyError as error:
    print("unknown option:", error.args[0])
    sys.exit()

info = div2k.Info(scale=scale, subset=subset,
                  sampling=sampling, resolution=resolution)

dataset = div2k.Dataset(info=info,
                        data_dir="./.cache/data")

dataset.download()
