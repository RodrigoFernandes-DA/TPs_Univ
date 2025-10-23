import os

# Limite de tamanho aceito pelo GitHub (100 MB)
LIMITE_MB = 100
LIMITE_BYTES = LIMITE_MB * 1024 * 1024

def verificar_arquivos_grandes(caminho='.'):
    print(f"\n🔍 Verificando arquivos maiores que {LIMITE_MB} MB em: {os.path.abspath(caminho)}\n")

    arquivos_grandes = []

    for raiz, _, arquivos in os.walk(caminho):
        for arquivo in arquivos:
            caminho_arquivo = os.path.join(raiz, arquivo)
            try:
                tamanho = os.path.getsize(caminho_arquivo)
                if tamanho > LIMITE_BYTES:
                    arquivos_grandes.append((caminho_arquivo, tamanho))
            except (FileNotFoundError, PermissionError):
                # Ignora arquivos inacessíveis ou removidos durante a varredura
                continue

    if not arquivos_grandes:
        print("✅ Nenhum arquivo acima do limite encontrado!")
    else:
        print("⚠️  Arquivos acima de 100 MB:\n")
        for caminho_arquivo, tamanho in sorted(arquivos_grandes, key=lambda x: x[1], reverse=True):
            tamanho_mb = tamanho / (1024 * 1024)
            print(f"{tamanho_mb:8.2f} MB  —  {caminho_arquivo}")

if __name__ == "__main__":
    verificar_arquivos_grandes(".")