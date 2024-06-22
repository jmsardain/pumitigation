## DNN/MLP version of PUMitigation

Once you get your data in a csv file (code to be added to this git repo), the following is done:

```
## To train:
python train.py --train --outdir ./out --lr 0.0001 --epochs 300 --batch_size 64
```

ATTENTION: some hyperparameters are set by default, so you can change them in the command line.

For the testing, do the following:
```
python train.py --test
```

The testing for now only does plots at cluster classification level. The dataframe with untransformed variables needs to be called so that the jet info can be calculated correctly.

## Action Items

- [ ] Add code to get the csv files: train, val, test transformed, test untransformed
- [ ] Import the plotting code so that the plots can be made on the fly without saving then reading dataframe
- [ ] In the plotting code, check that the score transformation is applied correctly
- [ ] In the plotting code, check that there's a closure for jetRawE (i.e. jetRawE = sum clusterE)
