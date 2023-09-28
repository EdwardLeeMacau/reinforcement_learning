# diff 2 pkl

import argparse
import os
import pickle

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", type=str, required=True, help="Path to the src file",
    )
    parser.add_argument(
        "--dest", type=str, required=True, help="Path to the dest file",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursive compare",
    )
    return parser.parse_args()

def compare_file(src, dest) -> bool:
    with open(src, 'rb') as f:
        src = pickle.load(f)

    with open(dest, 'rb') as f:
        dest = pickle.load(f)

    return np.all(src == dest)

def main():
    args = parse_args()

    if not args.recursive:
        print(compare_file(args.src, args.dest))
        return

    # tests
    # ├── maze
    # │   ├── async_dynamic_programming
    # │   │   └── async_dynamic_programming.png
    # │   ├── iterative_policy_evaluation
    # │   │   ├── iterative_policy_evaluation.png
    # │   │   ├── policy.pkl
    # │   │   └── values.pkl
    # │   ├── maze.png
    # │   ├── maze.txt
    # │   ├── policy_iteration
    # │   │   ├── policy_iteration.png
    # │   │   ├── policy.pkl
    # │   │   └── values.pkl
    # │   └── value_iteration
    # │       ├── policy.pkl
    # │       ├── value_iteration.png
    # │       └── values.pkl
    # └── maze_large
    #     ├── async_dynamic_programming
    #     │   └── async_dynamic_programming.png
    #     ├── iterative_policy_evaluation
    #     │   ├── iterative_policy_evaluation.png
    #     │   ├── policy.pkl
    #     │   └── values.pkl
    #     ├── maze_large.png
    #     ├── maze_large.txt
    #     ├── policy_iteration
    #     │   ├── policy_iteration.png
    #     │   ├── policy.pkl
    #     │   └── values.pkl
    #     └── value_iteration
    #         ├── policy.pkl
    #         ├── value_iteration.png
    #         └── values.pkl
    for dirname in os.listdir(args.src):
        for opt in ('iterative_policy_evaluation', 'policy_iteration', 'value_iteration'):
            src = os.path.join(args.src, dirname, opt)
            dest = os.path.join(args.dest, dirname, opt)

            ret = True
            for filename in ('values.pkl', 'policy.pkl'):
                ret = ret and compare_file(os.path.join(src, filename), os.path.join(dest, filename))

            print(f"{dirname:>16s}.{opt:<32s}: {ret}")

if __name__ == "__main__":
    main()
