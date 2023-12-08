python ./data/build_train.py  \
    --data_name yelp  \
    --train_file ./data/yelp/train.txt  \
    --item_file ./data/yelp/item.txt  \
    --item_ids_file ./data/yelp/item_name.jsonl  \
    --output train_name.jsonl  \
    --output_dir ./data/yelp  \
    --seed 2022  \
    --tokenizer t5-small   

python ./data/build_train.py  \
    --data_name yelp  \
    --train_file ./data/yelp/valid.txt  \
    --item_file ./data/yelp/item.txt  \
    --item_ids_file ./data/yelp/item_name.jsonl  \
    --output valid_name.jsonl  \
    --output_dir ./data/yelp  \
    --seed 2022  \
    --tokenizer t5-small   