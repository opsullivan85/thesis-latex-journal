import re
import os

corl_files = ['corl/introduction.tex', 'corl/methodology.tex', 'corl/results.tex', 'corl/conclusions.tex']
main_files = []
for d in ['introduction', 'background', 'methodology', 'results', 'conclusions', 'header', 'appendices']:
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith('.tex'):
                main_files.append(os.path.join(root, file))

def extract_contexts(files, context_lines=2):
    contexts = {}
    for fn in files:
        if not os.path.exists(fn):
            continue
        with open(fn) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            matches = re.finditer(r'\\cite\{([^\}]+)\}', line)
            for match in matches:
                cites = [c.strip() for c in match.group(1).split(',')]
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                snippet = "".join(lines[start:end]).strip()
                
                for cite in cites:
                    if cite not in contexts:
                        contexts[cite] = []
                    contexts[cite].append(f"[{fn}:{i+1}]\n    " + snippet.replace("\n", "\n    "))
    return contexts

corl_contexts = extract_contexts(corl_files)
main_contexts = extract_contexts(main_files)

with open('cite_compare_wide.txt', 'w') as out:
    for cite, strings in corl_contexts.items():
        out.write(f"\n=========================================\n")
        out.write(f"CITATION: {cite}\n")
        out.write(f"=========================================\n")
        out.write(">>> CORL USAGE:\n")
        for s in strings:
            out.write(f"  {s}\n\n")
        out.write(">>> MAIN USAGE:\n")
        if cite in main_contexts:
             for s in main_contexts[cite]:
                 out.write(f"  {s}\n\n")
        else:
             out.write("  NOT FOUND IN MAIN\n\n")

print("Done. Output written to cite_compare_wide.txt")
