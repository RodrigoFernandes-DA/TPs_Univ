Travaux Pratiques - Master Sciences des Donnees

#### Mudar de Diretorio ############################
cd /home/ENT-UR/AUTOFS/STUDENTS/coelhrod/Git
cd "/home/ENT-UR/AUTOFS/STUDENTS/coelhrod/Git/ML Sequence/4-TP-CTC"

#### Apagar do historico
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch 'ML Graph/TP_GNN_challenge_M2/dataset/ZFR/processed/data_train.pt'" --prune-empty --tag-name-filter cat -- --all

#### Varificar se apagou
git log --all -- "ML Graph/TP_GNN_challenge_M2/dataset/ZFR/processed/data_train.pt"

#### Senao, limpar do cache
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive


#### Pull forçado ################################
git checkout main
git fetch origin
git reset --hard origin/main


#### Ativar e desativar venv ####################
source venvTP/bin/activate
deactivate
rm -rf venv

#### pip install ################################
python3.12 -m pip install --user tqdm==4.66.4 --break-system-packages





https://learngerman.dw.com/en/learn-german/s-9528 # Videos
https://dartdrill.dartmouth.edu/driller # Exercicios
https://germanforenglishspeakers.com/ # Gramatica
https://german.net/exercises/tenses/ # Exercicios
https://www.vhs-lernportal.de/wws/9.php#/wws/home.php?sid=91614352505798270961022922292450Sa7fc939a
https://www.deutschakademie.de/online-deutschkurs/deutschkurs