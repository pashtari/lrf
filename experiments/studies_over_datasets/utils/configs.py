import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compression comparison over datasets")
    parser.add_argument("--data_dir", default="data/kodak_sub", type=str)
    parser.add_argument("--experiment_name", default="kodak", type=str)
    parser.add_argument("--patch_size", default=8, type=tuple)
    parser.add_argument("--bounds", default=16, type=tuple)
    parser.add_argument("--num_iters", default=10, type=int)
    parser.add_argument("--pachify", default=True, type=bool)
    parser.add_argument("--selected_methods", default=["JPEG","SVD","IMF-RGB","IMF-YCbCr"], type=list)
    args = parser.parse_args()
    return args