conda create -n mtbbench python=3.12

conda activate mtbbench

conda run -n mtbbench pip install numpy
conda run -n mtbbench pip install pandas
conda run -n mtbbench pip install einops
conda run -n mtbbench pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n mtbbench pip install biopython
conda run -n mtbbench pip3 install -U scikit-learn
conda run -n mtbbench pip install -U matplotlib
conda run -n mtbbench pip install seaborn
conda run -n mtbbench pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
conda run -n mtbbench pip install wandb
conda run -n mtbbench pip install pillow
conda run -n mtbbench pip install umap-learn
conda run -n mtbbench pip install POT
conda run -n mtbbench pip install loguru
conda run -n mtbbench pip install omegaconf imblearn
conda run -n mtbbench pip install flash-attn --no-build-isolation
conda run -n mtbbench pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
conda run -n mtbbench pip install -r requirements.txt
conda run -n mtbbench pip install scikit-image accelerate