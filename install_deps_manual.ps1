# Manual dependency installation for Python 3.11 venv
# Run this AFTER activating the virtual environment

Write-Host "Installing dependencies for VQ Encoder Decoder (Python 3.11)" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan

# Ensure venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "[ERROR] Virtual environment not activated!" -ForegroundColor Red
    Write-Host "Please run: .\vqvae_env\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[1] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

Write-Host "`n[2] Installing PyTorch 2.1.0 (compatible with torch-cluster)..." -ForegroundColor Yellow
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

Write-Host "`n[3] Installing PyG extensions (torch-scatter, torch-sparse, torch-cluster)..." -ForegroundColor Yellow
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

Write-Host "`n[4] Installing torch-geometric..." -ForegroundColor Yellow
python -m pip install torch-geometric

Write-Host "`n[5] Installing training dependencies..." -ForegroundColor Yellow
python -m pip install accelerate tqdm tensorboard pyyaml python-box

Write-Host "`n[6] Installing data processing dependencies..." -ForegroundColor Yellow
python -m pip install h5py biopython graphein omegaconf hydra-core einops torchmetrics

Write-Host "`n[7] Installing cosine annealing warmup..." -ForegroundColor Yellow
python -m pip install git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup.git

Write-Host "`n[8] Verifying installation..." -ForegroundColor Yellow
$testScript = @"
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
import torch_geometric
print(f'PyG: {torch_geometric.__version__}')
import torch_cluster
print('torch_cluster: OK')
import torch_scatter
print('torch_scatter: OK')
import torch_sparse
print('torch_sparse: OK')
import numpy
print(f'NumPy: {numpy.__version__}')
import accelerate
print(f'Accelerate: {accelerate.__version__}')
print('\n[SUCCESS] All packages installed!')
"@

python -c $testScript

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" + "=" * 80 -ForegroundColor Green
    Write-Host "INSTALLATION COMPLETE!" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "`nYou can now run training with:" -ForegroundColor Cyan
    Write-Host "    python train.py --config_path ./configs/config_vqvae.yaml" -ForegroundColor White
} else {
    Write-Host "`n[ERROR] Installation verification failed" -ForegroundColor Red
}
