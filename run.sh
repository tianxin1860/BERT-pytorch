#
#python -m pdb bert_pytorch -c train_data -v vocab_dict_utf8 -o output -w 1 -e 5 -b 3
#python -m bert_pytorch -c train_data -v vocab_dict_utf8 -o output -w 1 -e 3 -b 3
python -m bert_pytorch -c train_data_mini -v vocab_dict_utf8 -o output -w 1 -e 3 -b 25 --log_freq 1
