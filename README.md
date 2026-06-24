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

## convert fine tuned model to .gguf file (to use in lm-studio)
1. run merge_model.py in /helpers
2. build llama.cpp submodule (with make or cmake)
3. run ``` python convert_hf_to_gguf.py ../merged_model --outfile ../custom-model.gguf --outtype f16``` (whilst in llama.cpp folder)
4. to quantize the model run ```.\build\bin\Release\llama-quantize.exe ../custom-model.gguf ../custom-model-Q4_K_M.gguf Q4_K_M ```
5. then create custom folder structure ```user/.cache/lm-studio/models/PUBLISHER/CUSTOM-MODEL/CUSTOM_MODEL.gguf```
6. run in lm-studio