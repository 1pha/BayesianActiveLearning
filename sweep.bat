python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=bilstm --output_dir=output/init_pct/001_bilstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=bilstm --output_dir=output/init_pct/005_bilstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=bilstm --output_dir=output/init_pct/010_bilstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=bilstm --output_dir=output/init_pct/020_bilstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=bilstm --output_dir=output/init_pct/050_bilstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=bilstm --output_dir=output/init_pct/100_bilstm --learning_rate=1e-4

python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=cnnlstm  --output_dir=output/init_pct/001_cnnlstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=cnnlstm  --output_dir=output/init_pct/005_cnnlstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=cnnlstm  --output_dir=output/init_pct/010_cnnlstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=cnnlstm --output_dir=output/init_pct/020_cnnlstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=cnnlstm --output_dir=output/init_pct/050_cnnlstm --learning_rate=1e-4
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=cnnlstm --output_dir=output/init_pct/100_cnnlstm --learning_rate=1e-4

python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=bert  --output_dir=output/init_pct/001_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=bert  --output_dir=output/init_pct/005_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=bert  --output_dir=output/init_pct/010_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=bert --output_dir=output/init_pct/020_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=bert --output_dir=output/init_pct/050_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=bert --output_dir=output/init_pct/100_bert