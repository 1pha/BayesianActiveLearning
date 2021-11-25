python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/001_bilstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/005_bilstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/010_bilstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/020_bilstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/050_bilstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=bilstm --num_train_epochs=150 --intermediate_size=256 --output_dir=output/init_pct/100_bilstm

python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=cnnlstm --num_train_epochs=150  --output_dir=output/init_pct/001_cnnlstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=cnnlstm --num_train_epochs=150  --output_dir=output/init_pct/005_cnnlstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=cnnlstm --num_train_epochs=150  --output_dir=output/init_pct/010_cnnlstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=cnnlstm --num_train_epochs=150 --output_dir=output/init_pct/020_cnnlstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=cnnlstm --num_train_epochs=150 --output_dir=output/init_pct/050_cnnlstm
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=cnnlstm --num_train_epochs=150 --output_dir=output/init_pct/100_cnnlstm

python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.01 --model_name_or_path=bert --num_train_epochs=150  --output_dir=output/init_pct/001_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.05 --model_name_or_path=bert --num_train_epochs=150  --output_dir=output/init_pct/005_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.10 --model_name_or_path=bert --num_train_epochs=150  --output_dir=output/init_pct/010_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.20 --model_name_or_path=bert --num_train_epochs=150 --output_dir=output/init_pct/020_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=0.50 --model_name_or_path=bert --num_train_epochs=150 --output_dir=output/init_pct/050_bert
python code/run.py --do_train=True --do_valid=True --do_test=True --init_pct=1.00 --model_name_or_path=bert --num_train_epochs=150 --output_dir=output/init_pct/100_bert