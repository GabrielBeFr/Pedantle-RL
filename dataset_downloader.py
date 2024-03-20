from datasets import load_dataset
import pandas as pd

def main():
    dataset = load_dataset("wikipedia", "20220301.en")
    df = pd.DataFrame(dataset['train'])
    df.to_csv('data/wikipedia_dataset.csv', index=False)

if __name__ == "__main__":
    main()