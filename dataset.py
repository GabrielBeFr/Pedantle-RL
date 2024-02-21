from datasets import load_dataset
from pdb import set_trace

def create_wikipedia_dataset(split='train', cache_dir=None):
    """
    Function to create a dataset of Wikipedia articles.
    
    Args:
        language (str): The language of the Wikipedia articles. Default is 'en' (English).
        split (str): The split of the dataset to load. Default is 'train'.
        cache_dir (str): Directory to cache the downloaded dataset. Default is None.
    
    Returns:
        dataset: The dataset of Wikipedia articles.
    """
    dataset = load_dataset('wikipedia', '20220301.fr', split=split)
    return dataset

def get_wikipedia_articles(dataset, num_articles=10):
    """
    Function to get a specified number of Wikipedia articles from the dataset.
    
    Args:
        dataset: The dataset of Wikipedia articles.
        num_articles (int): The number of articles to retrieve. Default is 10.
    
    Returns:
        list: A list of dictionaries, where each dictionary represents an article.
    """
    articles = dataset[:num_articles]
    return articles


if __name__ == '__main__':
    dataset = create_wikipedia_dataset
    #set_trace()
    article = get_wikipedia_articles(dataset, num_articles=1)
    with open("article_visu.txt",'a') as f:
        for line in article:
            f.read(article)