from datasets import load_dataset
import pandas as pd
import wikipedia

WIKI_ARTICLES = ["Asteroid", "Lagos", "Artic Ocean", "Vasco de Gama", "Finance", "Weapon", "Humour", "Early human migration", "Chemistry", "Wheel", "Electric motor", "History of medicine", "Reptiles", "Johann Sebastian Bach", "Hospital", "Classical Mechanics", "Sanitation", "Argentina", "Confucius", "Communication", "Photosynthesis", "Suffrage", "Metaphysics", "Mediterranean Sea", "United Arab Emirates", "Garden", "Human behavior", "Birth Control", "Antartica", "Cardiovascular disease", "Plant", "Man", "Beijing", "God", "Tokyo", "Ferdinand Magellan", "Construction"]

def main():
    dataset = []
    for i, article in enumerate(WIKI_ARTICLES):
        try:
            page = wikipedia.page(article)
        except:
            print(f"Error: {article}")
            continue
        content = page.content.split("==")[0]
        data = {
            "id": i,
            "url": page.url,
            "title": page.title,
            "text": content
        }
        dataset.append(data)
    
    df = pd.DataFrame(dataset)
    df.to_csv("data/wikipedia_dataset.csv", index=False)

if __name__ == "__main__":
    main()