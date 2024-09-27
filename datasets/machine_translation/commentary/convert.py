file = "news-commentary-v11.fr-en.xliff"
sources = []
targets = []

with open(file, "r") as f:
    for line in f:
        if line[0:8] == "<source>":
            sources.append(line.strip()[8:-9])
        elif line[0:8] == "<target>":
            targets.append(line.strip()[8:-9])

assert len(sources) == len(targets)

output_name = "news-commentary-v11.fr-en"

with open(output_name + ".en", "w") as en, open(output_name + ".fr", "w") as fr:
    for en_line, fr_line in zip(targets, sources):
        en.write(en_line + "\n")
        fr.write(fr_line + "\n")
