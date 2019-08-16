src="/home/t/project_model/base/results/pol/democratic_only.test.en"
tgt="/home/t/project_model/base/results/pol/democratic_only.test.to.republican.1"
clear
cd ~/project_model/base/
python3 measure_ts.py -src $src -tgt $tgt -type political 
