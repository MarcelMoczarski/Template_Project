from pathlib import Path

filename = "test"
print(Path(__file__).resolve().parents[1]/"tmp_files"/filename)
