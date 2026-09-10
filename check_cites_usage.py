import re
import os

corl_files = ['corl/introduction.tex', 'corl/methodology.tex', 'corl/results.tex', 'corl/conclusions.tex']
main_files = []
for d in ['introduction', 'background', 'methodology', 'results', 'conclusions', 'header', 'appendices']:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.tex'):
                main_files.append(os.path.join(root, file))

def extract_contexts(files):
    contexts = {}
    for fn in files:
        if not os.path.exists(fn):
            continue
        with open(fn) as f:
            for i, line in enumerate(f):
                matches = re.finditer(r'\\cite\{([^\}]+)\}', line)
                for match in matches:
                    cites = [c.strip() for c in match.group(1).split(',')]
                    for cite in cites:
                        if cite not in contexts:
                            contexts[cite] = []
                        contexts[cite].append(f"[{fn}:{i+1}] {line.strip()}")
    return contexts

corl_contexts = extract_contexts(corl_files)
main_contexts = extract_contexts(main_files)

for cite, strings in corl_contexts.items():
    print(f"\n--- Citation: {cite} ---")
    print("CORL USAGE:")
    for s in strings:
        print("  " + s)
    print("MAIN USAGE:")
    if cite in main_contexts:
         for s in main_contexts[cite]:
             print("  " + s)
    else:
         print("  NOT FOUND IN MAIN")
