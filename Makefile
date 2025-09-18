train:
	uv run src/train.py --data data/dataset.yaml --epochs 50 --img 640

test:
	pytest tests/

run-notebook:
	uv jupyter notebook notebook/train_yolov8_signal_for_help.ipynb
