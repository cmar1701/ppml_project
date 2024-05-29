import pandas as pd


def main():
    df = pd.read_csv("data/HAM10000_metadata.csv")
    print(f"Describe: \n{df.describe()}")
    print(f"Shape: \n{df.shape}")
    print(f"Head: \n{df.head()}")

if __name__ == "__main__":
    main()
    