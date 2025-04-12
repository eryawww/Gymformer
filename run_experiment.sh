source scripts/download_openai_data.sh
python scripts/decode_openai_data.py
poetry run python launch.py --train-reward
poetry run python launch.py --train-lm