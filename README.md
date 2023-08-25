# PUMitigation

Let's try getting all the clusters that are coming from pileup!

## Data

Somewhere on /eos/.

## Idea

We should build a graph with the nodes/clusters insides jets and not in all the events: to do that, we should rely on the jetCnt variable (not done in the code now).

Clusters from pileup are characterized by a reconstructed energy (clusterE) that is >0 and a true energy (cluster\_ENG\_TOT\_CALIB == 0 ).
This will help us define 2 classes that should be defined using cluster\_ENG\_TOT\_CALIB:
```
PU cluster:     cluster\_ENG\_TOT\_CALIB == 0 && clusterE > 0
Non PU cluster: cluster\_ENG\_TOT\_CALIB > 0  && clusterE > 0
```

## Node features

For node features, we should use :

```
'clusterE',
'clusterEta',
'cluster_CENTER_LAMBDA',
'cluster_CENTER_MAG',
'cluster_ENG_FRAC_EM',
'cluster_FIRST_ENG_DENS',
'cluster_LATERAL',
'cluster_LONGITUDINAL',
'cluster_PTD',
'cluster_time',
'cluster_ISOLATION',
'cluster_SECOND_TIME',
'cluster_SIGNIFICANCE',
'nPrimVtx',
'avgMu'
```
