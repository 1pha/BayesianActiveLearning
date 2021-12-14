@REM python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --output_dir=output/active/cnnlstm_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500

@REM echo code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
@REM python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
@REM python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --output_dir=output/active/bert_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500
@REM 
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500 
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_random --active_learning=True --acquisition=random --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_lc --active_learning=True --acquisition=lc --acquisition_period=5 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_margin --active_learning=True --acquisition=margin --acquisition_period=5 --load_size=base --increment_num=1500

python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bilstm --retrain=True --output_dir=output/active_retrain/bilstm_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500

python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=cnnlstm --retrain=True --output_dir=output/active_retrain/cnnlstm_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500

python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_random --active_learning=True --acquisition=random --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_lc --active_learning=True --acquisition=lc --acquisition_period=10 --load_size=base --increment_num=1500
python code/run.py --do_train=True --do_valid=True --model_name_or_path=bert --retrain=True --output_dir=output/active_retrain/bert_margin --active_learning=True --acquisition=margin --acquisition_period=10 --load_size=base --increment_num=1500