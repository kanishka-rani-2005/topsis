import sys
from .core import run_topsis, TopsisError


def main():
    if len(sys.argv) != 5:
        print("Error: Incorrect number of parameters")
        print("Usage: topsis <InputDataFile> <Weights> <Impacts> <OutputResultFileName>")
        sys.exit(1)

    _, input_file, weights, impacts, output_file = sys.argv

    try:
        run_topsis(input_file, weights, impacts, output_file)
        print(f"Result saved to: {output_file}")
    except TopsisError as e:
        print(f"Error: {e}")
        sys.exit(1)