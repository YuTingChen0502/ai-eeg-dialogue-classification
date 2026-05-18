# Reproduce Final Part 2 Pipeline

Run from:

```powershell
cd C:\Coding\Intro_to_AI\project1-part2\part2
```

All commands use the dedicated CUDA Python executable:
`C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe`

```powershell
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" -c "import sys; print(sys.version)"
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" -c "import torch; print('cuda=', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" -c "import pandas, numpy, sklearn, transformers; print('core imports OK')"
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/deberta-v3-base'); print('DeBERTa tokenizer OK')"
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\preflight.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\heuristic_split.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\run_oof.py --model-name distilbert-base-uncased --model-tag distilbert_no_aug --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\run_oof.py --model-name distilbert-base-uncased --model-tag distilbert --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune --augment-heuristic-splitting
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\run_oof.py --model-name microsoft/deberta-v3-base --model-tag deberta_v3 --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\run_oof.py --model-name roberta-base --model-tag roberta --k-folds 5 --epochs 3 --batch-size 16 --max-length 128 --seed 42 --learning-rate 2e-5 --weight-decay 0.01 --threshold-tune
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\classical_oof.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\endpoint_features.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\retrieval_features.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\stack_oof.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\duplicate_transfer.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\summarize.py
& "C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe" scripts\update_notebook_summary.py
```
