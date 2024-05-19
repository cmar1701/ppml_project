import pandas as pd


def main():
    df = pd.read_csv("HAM10000_metadata.csv")
    df.describe()
    df.shape
    df.head()

if __name__ == "__main__":
    main()
    