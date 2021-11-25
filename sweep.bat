python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_lc_init10 --checkpoint_period=1 --active_learning=True --acquisition=lc --acquisition_period=5 --init_pct=0.1
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_margin_init10 --checkpoint_period=1 --active_learning=True --acquisition=margin --acquisition_period=5 --init_pct=0.1
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_random_init10 --checkpoint_period=1 --active_learning=True --acquisition=random --acquisition_period=5 --init_pct=0.1

python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_lc_init20 --checkpoint_period=1 --active_learning=True --acquisition=lc --acquisition_period=5 --init_pct=0.2
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_margin_init20 --checkpoint_period=1 --active_learning=True --acquisition=margin --acquisition_period=5 --init_pct=0.2
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_random_init20 --checkpoint_period=1 --active_learning=True --acquisition=random --acquisition_period=5 --init_pct=0.2

python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_lc_init30 --checkpoint_period=1 --active_learning=True --acquisition=lc --acquisition_period=5 --init_pct=0.3
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_margin_init30 --checkpoint_period=1 --active_learning=True --acquisition=margin --acquisition_period=5 --init_pct=0.3
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_random_init30 --checkpoint_period=1 --active_learning=True --acquisition=random --acquisition_period=5 --init_pct=0.3

python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_lc_init50 --checkpoint_period=1 --active_learning=True --acquisition=lc --acquisition_period=5 --init_pct=0.5
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_margin_init50 --checkpoint_period=1 --active_learning=True --acquisition=margin --acquisition_period=5 --init_pct=0.5
python code/run.py --do_train=True --do_valid=True --do_test=True --output_dir=output/active_learning/bert_random_init50 --checkpoint_period=1 --active_learning=True --acquisition=random --acquisition_period=5 --init_pct=0.5
 

