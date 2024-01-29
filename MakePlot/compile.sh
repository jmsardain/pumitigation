#!/bin/bash

COMPILER=$(root-config --cxx)
FLAGS=$(root-config --cflags --libs)
echo $COMPILER $FLAGS

$COMPILER $FLAGS -g -O3 -Wall -Wextra -Wpedantic -fopenmp ./doPlot.cc ./Plot.C ./AtlasStyle.C ./AtlasUtils.C ./AtlasLabels.C -I. -o doPlot
