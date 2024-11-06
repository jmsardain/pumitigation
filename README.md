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

For plotting, the code HistoMaker.C makes sure to get the observables per jet. In a clean new terminal:
```
setupATLAS
lsetup "root 6.30.02-x86_64-centos7-gcc11-opt"
cd MakePlot/
source compile.sh
./doPlot /path/to/ROOTFILE/in/ckpts/out_model.root
```

N.B.: the code using the TLorentzVector for now is giving the same values for jetPU and jetEM, either bug in the way HistoMaker.C is handling the values, or problem in the pumitigation code.

The Plotting.ipynb notebook takes care of the final plotting


## Plots and info for the PUBNote 

- [ ] Control plots for all features
- [ ] List of hyperparameters (batch size, learning rate) 
- [ ] Plot of loss function vs epoch for training and validation 
- [ ] Plot for jet response vs pT and jet resolution vs pT  
