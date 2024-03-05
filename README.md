First you need to make the ROOT files. Use MakeROOT: 
```
setupATLAS
lsetup "root 6.30.02-x86_64-centos7-gcc11-opt" 
source compile.sh
./doTree calib
./doTree pu
```

Second make data for calibration and train calibration: 

```
python3 train_calibration.py --train
python3 train_calibration.py --retrain
```

Third, make data for pytorch and train graph:
```
source /data/jmsardain/LJPTagger/JetTagging/miniconda3/bin/activate
conda activate rootenv

python makeData_pumitigation.py
python train_pumitigation.py config_EdgeConv.yml
python do_output.py config_test_EdgeConv.yml
```

