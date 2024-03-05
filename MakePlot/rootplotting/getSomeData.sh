#!/usr/bin/env bash

# Source- and target directories for test files
SOURCE_STABLE=/eos/atlas/user/a/asogaard/Analysis/2016/BoostedJetISR/outputObjdef/2017-04-27
SOURCE_LIGHT=/afs/cern.ch/user/a/asogaard/Analysis/2016/BoostedJetISR/AnalysisTools/outputObjdef
TARGET=data

if [[ `echo $HOSTNAME | cut -c1-6` == "lxplus" ]]; then
    # Running on lxplus; simply create symbolic link
    rm -f ./$TARGET
    ln -s $SOURCE_STABLE $TARGET

else
    # Running elsewhere; copy files
    UNAME=
    
    if [ ! $UNAME ]; then
	echo "Not running on lxplus; please provide a username (UNAME)"
    else
	mkdir -p $TARGET
	scp -r $UNAME@lxplus.cern.ch:$SOURCE_LIGHT/objdef_MC_3610*.root  ./$TARGET/
	scp -r $UNAME@lxplus.cern.ch:$SOURCE_LIGHT/objdef_MC_30543*.root ./$TARGET/
	scp -r $UNAME@lxplus.cern.ch:$SOURCE_LIGHT/objdef_MC_30544*.root ./$TARGET/
    fi
    
fi