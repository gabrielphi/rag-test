import os

files = [
    r"f:\rag-test\Self-Hosted-LLM\documentos\Sistema\raças\01_Altyra.txt",
    r"f:\rag-test\Self-Hosted-LLM\documentos\Sistema\raças\02_Ashka.txt"
]

for fpath in files:
    print(f"\n--- FILE: {os.path.basename(fpath)} ---")
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            # Print first 500 chars and any lines starting with #
            print(content[:500])
            print("\n... (HEADERS FOUND):")
            for line in content.splitlines():
                if line.strip().startswith('#'):
                    print(line.strip())
    else:
        print("File not found.")
