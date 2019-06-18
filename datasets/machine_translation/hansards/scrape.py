import os

"""
quick script to go through the hansards files.
"""

# you'll want to have a recursive browser.
def snoop(path):
    ents = []
    entries = os.listdir(path)
    if len(entries) > 5:
        ents.append(path)
    for ent in entries:
        direct_path = os.path.join(path,ent)
        if os.path.isdir(direct_path):
            ents += snoop(direct_path)
    return ents

# get the folders that contain the relevant files.
paths = snoop("hansard.36")

# this dictionary is structured as such `en` is the key, and `fr` is the value.
corpus = {}

# since this is a french-english corpus we can hardcode the values no problem.
enc = "ISO-8859-1"
for folder in paths:
    entries = set([filename[:-2] for filename in os.listdir(folder)])
    for entry in entries:
        entry = os.path.join(folder, entry)
        fr_file, en_file = entry + ".f", entry + ".e"
        # print(fr_file)
        with open(fr_file, encoding=enc) as f, open(en_file, encoding=enc) as r:
            for f_txt, e_txt in zip(f,r):
                f_txt, e_txt = f_txt.strip(), e_txt.strip()
                if len(f_txt) < 1 or len(e_txt) < 1:
                    continue
                """note: "Neither the sentence splitting nor the alignments are perfect. In particular, watch out for pairs that differ considerably in length. You may want to filter these out before you do any statistical training." -> https://www.isi.edu/natural-language/download/hansard/"""
                if len(f_txt)/len(e_txt) < 0.6 or len(e_txt)/len(f_txt) < 0.6:
                    continue
                corpus[e_txt] = f_txt

# now we need to save these files.
filename = "hansards"
filepath = filename + "."
p = 'w' # <-- permissions.
with open(filepath+"en", p) as e, open(filepath+"fr", p) as f:
    for e_txt in corpus:
        e.write(e_txt+"\n")
        f.write(corpus[e_txt]+"\n")