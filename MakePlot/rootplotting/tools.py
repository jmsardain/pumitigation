# -*- coding: utf-8 -*-

""" Collection of utility tools.

@file:   tools.py
@date:   25 April 2017
@author: Andreas Søgaard
@email:  andreas.sogaard@cern.ch
"""

# Basic import(s)
import os

# Scientific import(s)
import ROOT
try:
    from root_numpy import tree2array

    import numpy as np
    from numpy.lib.recfunctions import append_fields
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source snippets/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise

# Global definitions
inf = np.finfo(float).max
eps = np.finfo(float).eps


def loadXsec (path, delimiter=',', comment='#'):
    """ Load cross section weights from file. """
    # @TODO: Use lambda's as accessors, to make to completely general?

    xsec = dict()
    with open(path, 'r') as f:
        for l in f:
            line = l.strip()
            if line == '' or line.startswith(comment):
                continue
            fields = [f.strip() for f in line.split(delimiter)]

            if fields[2] == 'Data':
                continue

            # @TEMP: Assuming sum-of-weights normalisation included in per-event MC weights
            xsec[int(fields[0])] = float(fields[1]) * float(fields[3])

            pass
        pass

    return xsec


def loadData (paths, tree, branches=None, start=None, stop=None, step=None, prefix=None):
    """ Load data from ROOT TTree. """

    # Initialise output
    output = None

    # Loop file paths
    for ipath, path in enumerate(paths):

        # Open ROOT file
        if not os.path.isfile(path):
            warning("File '{:s}' does not exist.".format(path))
            continue

        if not path.endswith('.root.__v1'):
            warning("File '{:s}' is not a ROOT file.".format(path))
            continue

        f = ROOT.TFile(path, 'READ')

        # Get TTree
        t = f.Get(tree)

        if type(t) is not ROOT.TTree:
            warning("TTree '{:s}' was not found in file '{:s}'.".format(tree, path))
            continue

        # Read TTree into numpy array
        use_branches = branches
        if None not in [prefix, branches]:
            use_branches = [prefix + br for br in branches] + ['weight']
            pass
        data = tree2array(t, branches=use_branches, start=start, stop=stop, step=step)

        # Append id field
        data = append_fields(data, 'id', np.ones((data.size,)) * ipath, dtypes=int)

        # Concatenate
        # Note: Force to be of tyoe numpy.ndarray, to avoid errors with mask when renaming
        if output is None:
            output = np.array(data, dtype=data.dtype)
        else:
            try:
                output = np.array(np.concatenate((output,data)), dtype=output.dtype)
            except TypeError:
                warning("Inconsistency in the types of the structured arrays being concatenated for path: " + path)
                existing = output.dtype.names
                new      = data  .dtype.names
                warning("Existing:")
                for name in existing:
                    warning("  '%s' %s" % (name, '<---' if name not in new else ''))
                    pass
                warning("New:")
                for name in new:
                    warning("  '%s' %s" % (name, '<---' if name not in existing else ''))
                    pass
                raise
            pass

        pass

    # Check(s)
    if output is None:
        warning("No data was loaded")
    else:
        # Remove prefix (opt.)
        if prefix:
            output.dtype.names = [name.replace(prefix, '') for name in output.dtype.names]
            pass
        pass

    return output


def scale_weights (data, info, xsec=None, lumi=None, verbose=False):
    """ Scale data array by (opt.) cross section x efficiency and (opt.) luminosity. """

    # Check(s)
    if xsec is None and lumi is None:
        warning("Neither cross section nor luminosity provided. Nothing to do here.")
        return data

    # Append DSID field to 'data' array
    data = append_fields(data, 'DSID', np.zeros((data.size,)), dtypes=int)
    data = append_fields(data, 'isMC', np.zeros((data.size,)), dtypes=bool)

    # Loop file indices in 'info' array
    for idx, id in enumerate(info['id']):

        if verbose:
            print "Processing index %d out of %d:" % (idx + 1, len(info['id']))
            pass

        # Get mask of all 'data' entries with same id, i.e. from same file
        msk = (data['id'] == id)

        # Get DSID/isMC for this file
        DSID = info['DSID'][idx]
        isMC = info['isMC'][idx]

        if verbose:
            print "-- Got DSID %d" % DSID
            print "-- Cross section available for this DSID? %s" % ("YES" if DSID in xsec else "NO")
            if DSID in xsec:
                print "-- Scaling by cross section:", xsec[DSID]
                pass
            pass

        # Scale by cross section x filter eff. for this DSID
        if DSID in xsec:
            data['weight'][msk] *= xsec[DSID]
        else:
            warning("DSID %d not found in cross section dict." % DSID)
            pass

        # Store DSID/isMC
        data['DSID'][msk] = DSID
        data['isMC'][msk] = isMC
        pass

    # Scale by luminosity
    if lumi is not None:
        data['weight'] *= lumi
        pass

    return data


def get_maximum (hist):
    """ Return the maximum bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return get_maximum(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = -inf, ROOT.Double(0), ROOT.Double(-inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            output = max(float(output),float(y))
            pass
        return output

    try:
        return max(map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)))
    except ValueError:
        warning("get_maximum: No bins were found.")
    except:
        warning("get_maximum: Something didn't work here, for intput:")
        print hist
        pass
    return None


def get_minimum (hist):
    """ Return the minimum bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return get_minimum(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = inf, ROOT.Double(0), ROOT.Double(inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            output = min(float(output),float(y))
            pass
        return output

    try:
        return min(map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)))
    except ValueError:
        warning("get_minimum: No bins were found.")
    except:
        warning("get_minimum: Something didn't work here, for intput:")
        print hist
        pass
    return None

def get_minimum_positive (hist):
    """ Return the minimum positive bin content for a histogram. Assumes ... . Throws error if ... .  """

    # Check(s)
    if type(hist) == ROOT.THStack:
        return inf if hist.GetNhists() == 0 else get_minimum_positive(hist.GetStack()[0])#get_minimum_positive(get_stack_sum(hist))
    elif type(hist) in [ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
        N = hist.GetN()
        output, x, y = inf, ROOT.Double(0), ROOT.Double(inf)
        for i in range(N):
            hist.GetPoint(i, x, y)
            if x <= 0: continue
            output = min(float(output),float(y))
            pass
        return output

    try:
        return min([m for m in map(hist.GetBinContent, range(1, hist.GetXaxis().GetNbins() + 1)) if m > 0])
    except ValueError:
        warning("get_minimum_positive: No bins were found.")
    except:
        warning("get_minimum_positive: Something didn't work here, for intput:")
        print hist
        pass
    return None


def get_stack_sum (stack, only_first=True):
    """ ... """

    # Kinda hacky...
    if only_first:
        sumHisto = stack.GetHists()[0].Clone('sumHisto')
    else:
        # @TODO: Errors are not being treated properly...
        sumHisto = None
        for hist in stack.GetHists(): ##stack.GetStack():
            if sumHisto is None:
                sumHisto = hist.Clone('sumHisto')
            else:
                sumHisto.Add(hist)
                pass
            pass
        pass
    return sumHisto


def is_overlay (pad):
    """ Determine whether input pad is of type 'overlay' """
    return type(pad).__name__.endswith('overlay')


def is_canvas (pad):
    """ Determine whether input pad is of type 'canvas' """
    return type(pad).__name__.endswith('canvas')


def warning (string):
    """ ... """
    print '\033[91m\033[1mWARNING\033[0m ' + string
    return


def snapToAxis (x, axis):
    """ ... """

    bin = axis.FindBin(x)
    if   bin <= 0:
        xdraw = axis.GetXmin()
    elif bin > axis.GetNbins():
        xdraw = axis.GetXmax()
    else:
        down   = axis.GetBinLowEdge(bin)
        up     = axis.GetBinUpEdge (bin)
        middle = axis.GetBinCenter (bin)

        # Assuming snap to nearest edge. # @TODO: Improve?
        d1 = abs(x - down);
        d2 = abs(x - up);
        if d1 == d2:
            warning("Coordinate exactly in the middle of bin. Returning lower edge.")
            pass
        if (d1 <= d2): xdraw = down
        else:          xdraw = up
        pass
    return xdraw


def wait ():
    """ Generic wait function.

    Halts the execution of the script until the user presses ``Enter``.
    """
    raw_input("\033[1mPress 'Enter' to continue...\033[0m")
    return
