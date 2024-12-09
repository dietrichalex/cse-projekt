# cse-projekt

## setup pyvenv
**💡 Note:** both probably done by pycharm
+ create venv with -> `python -m venv venv`
+ activate venv with -> `.\venv\Scripts\Activate.ps1` <br>

**❗ Important:** has to be done manually <br>
+ install dependencies with -> `pip install -r .\config\requirements.txt` <br>

## setup git hooks
+ go to folder `.\config\hooks`
+ copy both `hook` files and paste them into the `.git/hooks/` folder
+ **❗ Important:** remove `".sample"` ending

## git commands
PUSHING AND COMMITING: <br>
+ git status -> current status
+ git add -A -> adds all to commit
+ git commit -m "message" -> commits and adds message to  commit (if you want to change message after already commiting use git commit --amend (💡 USES VIM))
+ git push

FETCHING AND PULLING: <br>
+ git fetch -p
+ git pull

SWITCH BRANCH: <br>
+ git checkout name_of_branch

## enable long paths
+ execute in PowerShell:
```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" 
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## cuda setup
+ install cuda 11.8
+ install torch
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
