import ast
import os
import glob
import subprocess
from collections import Counter

# Bibliotecas padrão do Python - NÃO incluir no requirements
STDLIB_MODULES = {
    'os', 'sys', 'math', 'json', 'csv', 're', 'datetime', 'time', 
    'collections', 'itertools', 'functools', 'random', 'statistics',
    'argparse', 'logging', 'pathlib', 'typing', 'unittest', 'pickle',
    'socket', 'threading', 'multiprocessing', 'subprocess', 'hashlib',
    'base64', 'html', 'xml', 'urllib', 'ssl', 'zipfile', 'tarfile',
    'shutil', 'glob', 'fnmatch', 'tempfile', 'wave', 'audioop',
    'array', 'struct', 'copy', 'pprint', 'traceback', 'getpass',
    'curses', 'platform', 'errno', 'ctypes', 'mmap', 'select',
    'signal', 'atexit', 'imp', 'importlib', 'parser', 'symtable',
    'token', 'tokenize', 'tabnanny', 'py_compile', 'pyclbr',
    'linecache', 'ast', 'dis', 'inspect', 'site', 'code', 'codeop',
    'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'threading',
    'dummy_threading', 'concurrent', 'multiprocessing', 'subprocess',
    'sched', 'queue', 'contextlib', 'decimal', 'fractions', 'random',
    'math', 'cmath', 'numbers', 'statistics', 'itertools', 'functools',
    'operator', 'collections', 'heapq', 'bisect', 'array', 'weakref',
    'copy', 'pprint', 'reprlib', 'enum', 'graphlib', 'typing'
}

def analisar_imports_arquivo(arquivo):
    """Analisa imports de um único arquivo"""
    imports = set()
    
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Análise AST para imports
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    lib_name = alias.name.split('.')[0]
                    if lib_name not in STDLIB_MODULES:
                        imports.add(lib_name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    lib_name = node.module.split('.')[0]
                    if lib_name not in STDLIB_MODULES:
                        imports.add(lib_name)
                        
    except Exception as e:
        print(f"Erro ao analisar {arquivo}: {e}")
    
    return imports

def analisar_projeto(caminho_projeto):
    """Analisa todos os arquivos Python do projeto"""
    todos_imports = set()
    
    # Padrões de arquivos para analisar
    padroes = ['**/*.py', '**/*.ipynb']
    
    for padrao in padroes:
        for arquivo in glob.glob(os.path.join(caminho_projeto, padrao), recursive=True):
            if arquivo.endswith('.ipynb'):
                # Converter notebook para Python temporariamente
                imports_notebook = analisar_ipynb(arquivo)
                todos_imports.update(imports_notebook)
            else:
                imports = analisar_imports_arquivo(arquivo)
                todos_imports.update(imports)
    
    return sorted(todos_imports)

def analisar_ipynb(arquivo_ipynb):
    """Analisa imports de notebooks Jupyter"""
    imports = set()
    
    try:
        import json
        with open(arquivo_ipynb, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        for cell in notebook.get('cells', []):
            if cell['cell_type'] == 'code':
                code = ''.join(cell['source'])
                
                # Análise simplificada para notebooks
                lines = code.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import '):
                        libs = line.replace('import ', '').split(',')
                        for lib in libs:
                            lib_name = lib.strip().split('.')[0]
                            if lib_name not in STDLIB_MODULES:
                                imports.add(lib_name)
                    
                    elif line.startswith('from '):
                        parts = line.split(' ')
                        if len(parts) >= 2:
                            lib_name = parts[1].split('.')[0]
                            if lib_name not in STDLIB_MODULES:
                                imports.add(lib_name)
                                
    except Exception as e:
        print(f"Erro ao analisar notebook {arquivo_ipynb}: {e}")
    
    return imports

def criar_pipfile(imports, caminho_projeto):
    """Cria Pipfile com as dependências encontradas"""
    pipfile_content = f'''[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
{"".join([f'{lib} = "*"\n' for lib in sorted(imports)])}

[dev-packages]

[requires]
python_version = "3.9"

[scripts]
start = "python main.py"
notebook = "jupyter notebook"
lab = "jupyter lab"
'''
    
    with open(os.path.join(caminho_projeto, 'Pipfile'), 'w', encoding='utf-8') as f:
        f.write(pipfile_content)
    
    print(f"Pipfile criado com {len(imports)} dependências")

# Uso
if __name__ == "__main__":
    projeto_path = input("Caminho do projeto: ").strip() or "."
    imports = analisar_projeto(projeto_path)
    
    print("\\n📦 Bibliotecas identificadas:")
    for i, lib in enumerate(imports, 1):
        print(f"{i:2d}. {lib}")
    
    criar_pipfile(imports, projeto_path)