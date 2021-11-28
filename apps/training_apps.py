import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataaccessframeworks.read_data import read_raw

def main():
    df = read_raw()

if __name__== "__main__":
    main()
