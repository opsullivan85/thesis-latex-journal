import re
import os

corl_cites = set()
for fn in ['corl/introduction.tex', 'corl/methodology.tex', 'corl/results.tex', 'corl/conclusions.tex']:
    if os.path.exists(fn):
        with open(fn) as f:
            for match in re.finditer(r'\\cite\{([^\}]+)\}', f.read()):
                for cite in [x.strip() for x in match.group(1).split(',')]:
                    corl_cites.add(cite)

print(f"Total citations in corl: {len(corl_cites)}")

main_cites = set()
main_cites_dict = {}
for d in ['introduction', 'background', 'methodology', 'results', 'conclusions', 'header', 'appendices']:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.tex'):
                path = os.path.join(root, file)
                with open(path) as f:
                    for match in re.finditer(r'\\cite\{([^\}]+)\}', f.read()):
                        for cite in [x.strip() for x in match.group(1).split(',')]:
                            main_cites.add(cite)
                            if cite not in main_cites_dict:
                                main_cites_dict[cite] = []
                            main_cites_dict[cite].append(path)

print(f"Total citations in main: {len(main_cites)}")

missing_in_main = corl_cites - main_cites
print(f"Citations in corl NOT in main: {missing_in_main}")

# Also let's check for inconsistent contexts...
