wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/tldr/online_45k.json -o ../data/tldr_online_45k.json
wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/tldr/offline_60k.json -o ../data/tldr_offline_60k.json
wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/cnndm/online_45k.json -o ../data/cnndm_online_45k.json
wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/cnndm/offline_60k.json -o ../data/cnndm_offline_60k.json
wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/descriptiveness/offline_5k.json -o ../data/descriptiveness_offline_5k.json
wget https://openaipublic.blob.core.windows.net/lm-human-preferences/labels/sentiment/offline_5k.json -o ../data/sentiment_offline_5k.json

# https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/book_passages/{mode}.jsonl
# https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/cnndm/url_lists/all_{mode}.txt
# https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/{mode}-subset.json