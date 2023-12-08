python ./data/gen_all_items.py  \
    --data_name yelp  \
    --item_file ./data/yelp/item.txt  \
    --output item_name.jsonl  \
    --output_dir ./data/yelp/  \
    --tokenizer t5-base   


# python gen_all_items.py  \
#     --data_name Amazon  \
#     --item_file /data1/meisen/TASTE-main/Data/sports/item.txt  \
#     --output item_name.jsonl  \
#     --output_dir /data1/meisen/TASTE-main/Data/sports  \
#     --tokenizer t5-base   

# python gen_all_items.py  \
#     --data_name Amazon  \
#     --item_file /data1/meisen/TASTE-main/Data/beauty/item.txt  \
#     --output item_name.jsonl  \
#     --output_dir /data1/meisen/TASTE-main/Data/beauty  \
#     --tokenizer t5-base  

# python gen_all_items.py  \
#     --data_name Amazon  \
#     --item_file /data1/meisen/TASTE-main/Data/toys/item.txt  \
#     --output item_name.jsonl  \
#     --output_dir /data1/meisen/TASTE-main/Data/toys  \
#     --tokenizer t5-base  