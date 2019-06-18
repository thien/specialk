import argparse
import os

desc = """
clean.py

deals with sanitising the global voices parallel corpus (moses format). Requires `argparse`. Makes the assumption that the corpus is already extracted. If you haven't done so, then you might want to do that first.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-folder_path', required=True, help="Default path for the extracted dataset file.")
parser.add_argument('-source', required=True, default="en",
                    help="Path to the training source data")
parser.add_argument('-target', required=True, default="fr",
                    help="Path to the training source data")
# parser.add_argument('-save_location', required=True)

opt = parser.parse_args()

# init directory
directory = os.listdir(opt.folder_path)

# make sure the locales exist.
possible_locales = set([i.split(".")[-1] for i in directory])
opt.source, opt.target = opt.source.lower(), opt.target.lower()
if not opt.source in possible_locales:
    print("The source is not in the possible locales. Please try one of the following:")
    print(possible_locales)
    exit
if not opt.target in possible_locales:
    print("The target is not in the possible locales. Please try one of the following:")
    print(possible_locales)
    exit

# build the filenames
source_filegroup = "globalvoices."
if len([i for i in directory if source_filegroup + opt.source + "-" + opt.target in i]) > 0:
    source_filegroup = source_filegroup + opt.source + "-" + opt.target
else:
    source_filegroup = source_filegroup +opt.target + "-" + opt.source

# double check that the files exist.
source_file = source_filegroup + "." + opt.source
target_file = source_filegroup + "." + opt.target 
assert source_file in directory
assert target_file in directory

# setup function to strip tags (used for english locale.)
tag = "&middot; Global Voices"
tag_index = len(tag)
def strip_tag(content, locale):
    # print(locale, content)
    if locale != "en":
        return content
    if tag[-6:] != content[-6:]:
        return content
    if content[-tag_index:] == tag:
        out = content[:len(content)-tag_index-1]
        return out
    return content

# initiate output container.
src, tgt = [], []

# load files
source_filepath = os.path.join(opt.folder_path, source_file)
target_filepath = os.path.join(opt.folder_path, target_file)
with open(source_filepath) as s, open(target_filepath) as t:
    for s_txt, t_txt in zip(s,t):
        # skip non-matching sequences
        if len(s_txt) < 1 or len(t_txt) < 1:
            continue
        # strip sequence
        s_txt, t_txt = s_txt.strip(), t_txt.strip()
        # the en locale is a bit troublesome because there is often
        # tags in article headers. We'll need to remove them.
        clean_s = strip_tag(s_txt, opt.source)
        clean_t = strip_tag(t_txt, opt.target)
        # store responses.
        src.append(clean_s)
        tgt.append(clean_t)

# make sure our sequences make any sense.
assert len(src) == len(tgt)

# now we need to save these files.
filename = "globalvoices"
filepath = filename + "."
p = 'w' # <-- permissions.
with open(filepath+opt.source, p) as s, open(filepath+opt.target, p) as t:
    for s_txt, t_txt in zip(src, tgt):
        s.write(s_txt+"\n")
        t.write(t_txt+"\n")
print("Done!")