echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

echo code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --output_dir=output/active/bilstm_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500

echo code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500

echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500