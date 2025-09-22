train:
	uv run src/train.py --data data/dataset.yaml --epochs 1 --img 640

test:
	uv run pytest tests/

run-notebook:
	uv jupyter notebook notebook/train_yolov8_signal_for_help.ipynb
