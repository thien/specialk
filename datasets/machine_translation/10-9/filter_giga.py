from tqdm import tqdm

output_en = "gigafren.en"
output_fr = "gigafren.fr"
src_en = "giga-fren.release2.fixed.en"
src_fr = "giga-fren.release2.fixed.fr"
seq_limit = 10000000

# don't edit from here.
total = 22520376  # precalculated

lims = {
    ".": 5,
    "(": 5,
    ")": 5,
    ",": 6,
    "@": 5,
}


def verify(eng):
    if "|" in eng:
        return False
    if "<" in eng or ">" in eng:
        return False
    if "^" in eng:
        return False
    if sum(c.isdigit() for c in eng) / len(eng) > 0.3:
        return False
    base = {}
    for char in eng:
        if char not in base:
            base[char] = 0
        base[char] += 1
        if char in lims:
            if lims[char] < base[char]:
                return False

    if "http" in eng.lower():
        return False
    return True


with open(output_en, "w", encoding="utf-8") as tgt_e, open(
    output_fr, "w", encoding="utf-8"
) as tgt_f:
    with open(src_en, encoding="utf-8") as src_e, open(
        src_fr, encoding="utf-8"
    ) as src_f:
        count = 0
        for e, f in tqdm(zip(src_e, src_f), total=total):
            if verify(e):
                tgt_e.write(e)
                tgt_f.write(f)
            count += 1
            if count > seq_limit:
                break
