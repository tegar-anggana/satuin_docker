import json

with open("new_quran_embeddings.json", "r", encoding="utf8") as f:
    quran_embeddings = json.load(f)
    
print(type(quran_embeddings))
surah_names = list(quran_embeddings.keys())
print(surah_names)