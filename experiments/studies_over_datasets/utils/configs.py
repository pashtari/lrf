import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compression comparison over datasets")
    parser.add_argument("--data_dir", default="/Users/pourya/Desktop/Remote/vsc/vsc_scratch/kodak", type=str)
    parser.add_argument("--experiment_name", default="test", type=str)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--bounds", default=16, type=int)
    parser.add_argument("--num_iters", default=10, type=int)
    parser.add_argument("--patchify", default=1, type=int)
    parser.add_argument("--color_space", default="YCbCr", type=str)
    parser.add_argument("--selected_methods", '--list', nargs='+', default=["IMF"])
    args = parser.parse_args()
    return args