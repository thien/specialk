import os
from tqdm import tqdm
quality = {"theguardian", "thetimes"}
popular = {"thesun", "mirror"}
groups = {"quality":quality, "popular":popular}

for group in tqdm(groups):
    with open(group + ".en", "w") as base:
        for publication_src in tqdm(groups[group], desc=group):
            filepath = os.path.join("srcs",publication_src + ".en")
            if not os.path.isfile(filepath):
                continue
            with open(filepath) as source:
                s = len(publication_src)
                for line in source:
                    base.write(group + line[s:].lower())