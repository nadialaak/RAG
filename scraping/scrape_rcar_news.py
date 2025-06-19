import requests
from bs4 import BeautifulSoup
import json
import os

url = "https://www.rcar.ma/actualites"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

articles = []

# Sélectionner tous les blocs d'actualités
for item in soup.find_all("div", class_="content-item"):
    # Titre (le lien contient aussi le texte du titre)
    link_tag = item.find("a", class_="link")
    if link_tag:
        title = link_tag.text.strip()
        link = "https://www.rcar.ma" + link_tag.get("href")
    else:
        title = ""
        link = ""

    # Résumé
    content_div = item.find("div", class_="content")
    summary = content_div.text.strip() if content_div else ""

    # Date
    date_tag = item.find("b", class_="dateField")
    date = date_tag.text.strip() if date_tag else ""

    articles.append({
        "titre": title,
        "lien": link,
        "resume": summary,
        "date": date
    })

# Assurer que le dossier existe
output_path = "data/actualites/rcar_news.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Sauvegarder les données dans un fichier JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print(f"{len(articles)} articles sauvegardés dans {output_path}")
