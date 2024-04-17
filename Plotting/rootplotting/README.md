# rootplotting
Scripts for producing ROOT plots using a matplotlib-like interface. 


## Setup

To setup and perform a first test of the package, you can do:
```
$ git clone git@github.com:asogaard/rootplotting.git
$ cd rootplotting
$ # If you're not running on lxplus, set your lxplus username (UNAME) in getSomeData.sh
$ source getSomeData.sh
$ python example.py
```


## Contents

This repository contains several small python files that will hopyfully make your ROOT-plotting life a bit easier. In particular:

* [ap](ap): Contains plotting classes, in particular the `pad`, `canvas`, and `overlay` classes that allow you to create ROOT plots using an interface similar to that of the popular python library matplotlib:
```
import ROOT
from rootplotting import ap

# ...

# data, bkg, WZ = structured numpy arrays

# Create canvas
c = ap.canvas()

# Draw stacked backgrounds
h_WZ  = c.stack(WZ ['m'], bins=bins, weights=WZ ['weight'], \
               fillcolor=ROOT.kAzure + 2, label='W(qq) + #gamma', scale=1.0)
h_bkg = c.stack(bkg['m'], bins=bins, weights=bkg['weight'], \
                fillcolor=ROOT.kAzure + 7, label='Incl. #gamma')
    
# Draw stats. error of stacked sum
h_sum  = c.getStackSum()
c.hist(h_sum, fillstyle=3245, fillcolor=ROOT.kGray+2, \
       linecolor=ROOT.kGray + 3, label='Stats. uncert.', option='E2')
    
# Draw data
h_data = c.plot(data['m'], bins=bins, markersize=0.8, label='Data')

# Add labels and text
c.xlabel('Signal jet mass [GeV]')
c.ylabel('Events')
c.text(["#sqrt{s} = 13 TeV,  L = 36.1 fb^{-1}",
        "Trimmed anti-k_{t}^{R=1.0} jets"], 
       qualifier='Simulation Internal')

# Configure y-axis scale
c.log(True)

# Draw legend
c.legend()

# Save and show plot
c.save('test.pdf')
c.show()
```
* [tools.py](tools.py): Contains some utility functions, e.g. to make the reading of ROOT TTrees into numpy arrays easier.
* [style.py](style.py): Style sheet for the ROOT plots, based on the ATLAS style recommendations.
* [example.py](example.py): An python script showing how to make pretty plots in just a few lines. Run as 
```
$ python example.py
```
Requires that you have downloaded data of the correct format using...
* [getSomeData.sh](getSomeData.sh): Set your lxplus username in the script (`UNAME=...`) and run
```
$ source getSomeData.sh
```
to download some `ROOT` files that can be used along with the example above.
* [sampleInfo.csv](sampleInfo.csv): CSV-file containing cross-sections and generator filter efficiencies for the Monte Carlo samples downloaded using the [getSomeData.sh](getSomeData.sh) script.


## Dependencies

The only non-standard dependencies should be ROOT and numpy. If you're on lxplus, you can set up the latter as

```
$ source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh
```
