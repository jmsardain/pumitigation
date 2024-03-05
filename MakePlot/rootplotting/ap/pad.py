# -*- coding: utf-8 -*-

""" Wrapper around ROOT TPad, handling plotting, labeling, text, and legend.

@file:   pad.py
@date:   26 April 2017
@author: Andreas Søgaard
@email:  andreas.sogaard@cern.ch
"""

# Basic import(s)
import time

# Scientific import(s)
import ROOT
try:
    import numpy as np
    from root_numpy import fill_hist, array2hist
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source snippets/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise

# Local import(s) -- not very pretty...
try:
    # Running from external directory as "from rootplotting import ap"
    from ..tools import *
except ValueError:
    # Running from 'rootplotting' as "import ap"
    from tools import *
    pass
try:
    from ..style import *
except ValueError:
    from style import *
    pass


# Enum class, for easy handling different plotting cases
def Enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

PlotType = Enum('plot', 'hist', 'hist2d', 'stack', 'graph')


# Class definition
class pad (object):
    """
    docstring for pad
    @TODO: Elaborate!
    """

    def __init__(self, base, coords):
        """ Constructor. """
        super(pad, self).__init__()

        # Check(s)
        #assert type(base) in [ap.canvas, ap.pad], "..."
        assert type(coords) in [list, tuple], "Pad coordinates must be of type list or tuple."
        assert len(coords) == 4, "Number of coordinates provided {} is not equal to 4.".format(len(coords))
        assert coords[0] < coords[2], "X-axis coordinates must be increasing ({:%.3f}, :%.3f})".format(coords[0], coords[2])
        assert coords[1] < coords[3], "X-axis coordinates must be increasing ({:%.3f}, :%.3f})".format(coords[1], coords[3])

        # Member variables
        # -- TPad-type
        self._base = base
        self._base._bare().cd()
        self._pad = ROOT.TPad('pad_{}_{}'.format(self._base._bare().GetName(), id(self)), "", *coords)
        self._coords = coords
        self._scale  = (1./float(coords[2] - coords[0]), 1./float(coords[3] - coords[1]))

        # -- Book-keeping
        self._primitives = list()
        self._entries = list()
        self._stack = None
        self._legends = list()
        self._children = list()
        self._oob_up   = None
        self._oob_down = None

        # -- Plotting cosmetics
        self._padding = 0.4
        self._log  = False
        self._logx = False
        self._xlim = None
        self._ylim = None
        self._ymin = None # For log-plots
        self._line  = None
        self._latex = None

        # Draw pad
        self._base._bare().cd()
        self._pad.Draw()
        self._base._bare().Update()
        return


    def __del__ (self):
        """ Destructor. """
        for p in self._primitives:
            del p
            pass
        del self._pad
        return


    # Decorators
    # ----------------------------------------------------------------

    # Make sure that we're always on the current pad
    def cd (func):
        def wrapper(self, *args, **kwargs):
            if hasattr(self._pad, 'cd'):
                self._pad.cd()
                pass
            return func(self, *args, **kwargs)
        return wrapper

    # Update pad upon completion of methdd
    def update (func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if hasattr(self._pad, 'Modified'):
                self._pad.Modified()
                self._pad.Update()
                pass
            return result
        return wrapper



    # Public plotting methods
    # ----------------------------------------------------------------

    def plot (self, data, **kwargs):
        """ ... """
        return self._plot(PlotType.plot, data, **kwargs)


    def hist (self, data, **kwargs):
        """ ... """
        return self._plot(PlotType.hist, data, **kwargs)


    def hist2d (self, data, **kwargs):
        """ ... """
        return self._plot(PlotType.hist2d, data, **kwargs)


    def stack (self, data, **kwargs):
        """ ... """
        return self._plot(PlotType.stack, data, **kwargs)


    def graph (self, data, **kwargs):
        """ ... """
        return self._plot(PlotType.graph, data, **kwargs)


    def ratio_plot (self, data, **kwargs):
        """ ... """
        return self._ratio_plot(PlotType.plot, data, **kwargs)

    def diff_plot (self, data, **kwargs):
        """ ... """
        return self._diff_plot(PlotType.plot, data, **kwargs)



    # Public accessor/mutator methods
    # ----------------------------------------------------------------

    def getStackSum (self):
        """ ... """

        # Check(s)
        if self._stack is None: return None
        return get_stack_sum(self._stack, only_first=False)


    @update
    def log (self, log=True):
        """ ... """

        # Check(s)
        assert type(log) == bool, "Log parameter must be a boolean"

        # Set log
        self._log = log
        self._update()
        return


    @update
    def logx (self, logx=True):
        """ ... """

        # Check(s)
        assert type(logx) == bool, "Logx parameter must be a boolean"

        # Set logx
        self._logx = logx
        self._update()
        return


    def logy (self, **kwargs):
        """ Alias method for 'log'. """
        return self.log(**kwargs)


    @update
    def xlim (self, *args):
        """ ... """

        # Check(s)
        if len(args) == 0: # Accessor
            if self._xlim is None:
                return self._pad.GetUxmin(), self._pad.GetUxmax()
            else:
                return self._xlim
                pass
            pass

        if type(args) == list and len(args) == 1:
            self.xlim(*args)
            return

        assert len(args) == 2, "X-axis limits have size {}, which is different from two as required.".format(len(xlim))

        # Store axis limits
        self._xlim = args
        self._update()
        return


    @cd
    @update
    def ylim (self, *args):
        """ ... """

        # Check(s)
        if len(args) == 0: # Accessor
            if self._ylim is None:
                return self._pad.GetUymin(), self._pad.GetUymax()
            else:
                return self._ylim
                pass
            pass

        if type(args) == list and len(args) == 1:
            self.ylim(*args)
            return

        assert len(args) == 2, "Y-axis limits have size {}, which is different from two as required.".format(len(ylim))

        # Store axis limits
        self._ylim = args
        self._update()
        return


    @update
    def ymin (self, ymin):
        """ ... """

        # Check(s)
        # ...

        # Store axis limits
        self._ymin = ymin
        self._update()
        return


    @update
    def padding (self, padding):
        """ ... """

        # Check(s)
        assert padding > 0, "Padding must be greater than zero; %.2f requested." % padding
        assert padding < 1, "Padding must be smaller than one; %.2f requested." % padding

        # Set padding
        self._padding = padding
        self._update()
        return



    # Public line-drawing methods
    # ----------------------------------------------------------------

    @cd
    def line (self, x1, y1, x2, y2, **kwargs):
        """ ... """

        # Check(s)
        self._line = ROOT.TLine()
        self._line.SetLineStyle(2)
        self._line.SetLineColor(ROOT.kGray + 3)
        self._style_line(self._line, **kwargs)

        # Draw line
        self._line.DrawLine(x1, y1, x2, y2)
        return


    def lines (self, coords, **kwargs):
        """ ... """

        for coord in coords:
            self.line(*coord, **kwargs)
            pass
        return


    def ylines (self, ys, **kwargs):
        """ ... """

        for y in ys:
            self.yline(y, **kwargs)
            pass
        return


    def xlines (self, xs, **kwargs):
        """ ... """

        for x in xs:
            self.xline(x, **kwargs)
            pass
        return


    @cd
    def yline (self, y, **kwargs):
        """ ... """

        xaxis = self._xaxis()
        xmin, xmax = self.xlim() # xaxis.GetXmin(), xaxis.GetXmax()
        self.line(xmin, y, xmax, y, **kwargs)
        return


    @cd
    def xline (self, x, snap=False, text=None, text_align='TL', **kwargs):
        """ ... """

        # Check(s)
        if self._log:
            warning("Calling 'xline' after 'log' will lead to problems in finding the correct lower end of the line. Please reverse the order.")
            pass

        xdraw = x
        if snap:
            xdraw = snapToAxis(x)
            pass

        ymin, ymax = self.ylim()
        if self._base._pads.index(self) == 0:
            ymax = max(map(get_maximum, self._primitives))
            pass
        self.line(xdraw, ymin, xdraw, ymax, **kwargs)
        print "Drawing line:", (xdraw, ymin, xdraw, ymax)

        if text is not None:
            # Default settings
            offset = 0.005 * (self.xlim()[1] - self.xlim()[0])
            angle = 270
            align = 11
            ydraw = ymax

            opts = {'textcolor': ROOT.kGray+1, 'textsize': 13}
            if 'linecolor' in kwargs:
                opts['textcolor'] = kwargs['linecolor']
                pass
            if 'textsize' in kwargs:
                opts['textsize'] = kwargs['textsize']


            if   'R' in text_align.upper():
                angle = 270
                pass
            elif 'L' in text_align.upper():
                offset *= -1
                angle = 90
            else:
                warning("Neither 'R' nor 'L' (horisontal) found in 'text_align'.")
                pass

            if   'T' in text_align.upper():
                if   'L' in text_align.upper():
                    align = align%10 + 30
                elif 'R' in text_align.upper():
                    align = align%10 + 10
                    pass
                pass
                ydraw = ymax
            elif 'M' in text_align.upper():
                align = align%10 + 20
                ydraw = (ymin + ymax) * 0.5
            elif 'B' in text_align.upper():
                if   'L' in text_align.upper():
                    align = align%10 + 10
                elif 'R' in text_align.upper():
                    align = align%10 + 30
                    pass
                ydraw = ymin + abs(offset)
            else:
                warning("Neither 'T', 'M', nor 'B' (vertical) found in 'text_align'")
                pass

            self.latex(text, xdraw + offset, ydraw, angle=angle, align=align, **opts)
            pass

        return xdraw



    # Public text/decoration methods
    # ----------------------------------------------------------------

    def xlabel (self, title):
        """ ... """

        try:
            self._xaxis().SetTitle(title)
        except AttributeError:
            # No axis was found
            pass
        return


    def ylabel (self, title):
        """ ... """

        try:
            self._yaxis().SetTitle(title)
        except AttributeError:
            # No axis was found
            pass
        return


    @cd
    @update
    def text (self, lines=[], qualifier='', ATLAS=True):
        """ ... """

        # Check(s)
        # ...

        # Create text instance
        t = ROOT.TLatex()

        # Compute drawing coordinates
        h = self._pad.GetWh() / self._scale[1] #* (1. - self._fraction) # @TODO: Improve
        w = self._pad.GetWw() / self._scale[0]

        offset = 0.05
        ystep = t.GetTextSize() / float(h) * 1.30
        scale = 1.#(w/float(h) if w > h else h/float(w))

        x =       self._pad.GetLeftMargin() + offset * scale
        y = 1.0 - self._pad.GetTopMargin()  - offset - t.GetTextSize() / float(h) * 1.0

        # Draw ATLAS line
        if ATLAS or qualifier:
            t.DrawLatexNDC(x, y, "{ATLAS}{qualifier}".format(ATLAS="#scale[1.15]{#font[72]{ATLAS}}#scale[1.05]{  }" if ATLAS else "", qualifier= "#scale[1.05]{%s}" % qualifier))
            y -= ystep * 1.30
            pass

        # Draw lines.
        for line in lines:
            t.DrawLatexNDC(x, y, line)
            y -= ystep;
            pass

        return


    @cd
    @update
    def latex (self, string, x, y, align=21, angle=0, NDC=False, **kwargs):
        """ ... """

        # Check(s)
        self._latex = ROOT.TLatex()
        self._latex.SetTextAlign(align)
        self._latex.SetTextAngle(angle)
        self._style_text(self._latex, **kwargs)

        # Draw line
        if NDC:
            self._latex.DrawLatexNDC(x, y, string)
        else:
            self._latex.DrawLatex(x, y, string)
            pass
        return


    @cd
    @update
    def legend (self, header=None, categories=None,
                xmin=None,
                xmax=None,
                ymin=None,
                ymax=None,
                width=0.32, # 0.32 / 0.28
                horisontal='R',
                vertical='T',
                reverse=False,
                sort=False):
        """ Draw legend on TPad. """

        # Check(s)
        N = len(self._get_all_entries())
        if (N == 0) and (categories is None or len(categories) == 0) and (header is None):
            return

        if len(self._legends) > 0:
            warning('A legend has already been constructed.')
            pass

        if reverse and sort:
            warning("Requesting reversed _and_ sorted legend. Will default to the former.")
            pass

        # Compute drawing coordinates
        h = self._pad.GetWh() / self._scale[1]
        w = self._pad.GetWw() / self._scale[0]

        fontsize = ROOT.gStyle.GetLegendTextSize()

        offset = 0.05
        height = (N + (1 if header else 0) + (len(categories) if categories else 0)) * fontsize / float(h) * 1.30

        # Setting x coordinates.
        if not (xmin or xmax):
            if   horisontal.upper() == 'R':
                xmax = 1. - self._pad.GetRightMargin() - offset
            elif horisontal.upper() == 'L':
                xmin = self._pad.GetLeftMargin() + offset
            else:
                warning("legend: Horisontal option '%s' not recognised." % horisontal)
                return
            pass
        if xmax and (not xmin):
            xmin = xmax - width
            pass
        if xmin and (not xmax):
            xmax = xmin + width
            pass

        # Setting y coordinates.
        if not (ymin or ymax):
            if   vertical.upper() == 'T':
                ymax = 1.0 - self._pad.GetTopMargin() - offset - fontsize / float(h) * 1.7
            elif vertical.upper() == 'B':
                ymin = self._pad.GetBottomMargin() + offset
            else:
                warning("legend: Vertical option '%s' not recognised." % vertical)
                return
            pass
        if ymax and (not ymin):
            ymin = ymax - height
            pass
        if ymin and (not ymax):
            ymax = ymin + height
            pass

        # Create legend
        self._legends.append(ROOT.TLegend(xmin, ymin, xmax, ymax))

        if header:
            self._legends[-1].AddEntry(None, header, '')
            pass

        # @TODO: Defer to parent pad, if 'overlay' (?)
        if reverse or sort:

            stored = list()

            # Data
            for (h,n,t) in self._get_all_entries():
                if type(h) == ROOT.THStack: continue
                if 'data' in n.lower():
                    self._legends[-1].AddEntry(h, n, t)
                    stored.append(h)
                    pass
                pass

            # Non-uncertainties
            if reverse:
                for (h,n,t) in reversed(self._get_all_entries()):
                    if type(h) == ROOT.THStack: continue
                    if h in stored: continue
                    if 'uncert' in n.lower() or 'stat.' in n.lower() or 'syst.' in n.lower(): continue
                    self._legends[-1].AddEntry(h, n, t)
                    stored.append(h)
                    pass
            elif sort:
                # Sorting: Data, filled histgorams, line histograms, uncertainties
                for (h,n,t) in reversed(sorted(self._get_all_entries(), key=lambda tup: tup[0].Integral())):
                    if type(h) == ROOT.THStack: continue
                    if h in stored: continue
                    if 'uncert' in n.lower() or 'stat.' in n.lower() or 'syst.' in n.lower() or ('f' not in t.lower()): continue
                    self._legends[-1].AddEntry(h, n, t)
                    stored.append(h)
                    pass
                for (h,n,t) in reversed(sorted(self._get_all_entries(), key=lambda tup: tup[0].Integral())):
                    if type(h) == ROOT.THStack: continue
                    if h in stored: continue
                    if 'uncert' in n.lower() or 'stat.' in n.lower() or 'syst.' in n.lower() or ('f' in t.lower()): continue
                    self._legends[-1].AddEntry(h, n, t)
                    stored.append(h)
                    pass
                pass

            # Rest (uncert.)
            for (h,n,t) in self._get_all_entries():
                if type(h) == ROOT.THStack: continue
                if h in stored: continue
                self._legends[-1].AddEntry(h, n, t)
                pass
        else:
            for (h,n,t) in self._get_all_entries():
                if type(h) == ROOT.THStack: continue
                self._legends[-1].AddEntry(h, n, t)
                pass
            pass

        # Add categories (opt.)
        if categories:
            for icat, (name, kwargs) in enumerate(categories):
                hist = ROOT.TH1F(name, "", 1, 0, 1)
                hist.SetBinContent(0, 1) # To avoid warning from get_minimum_positive
                self._add_to_primitives(hist)
                #self._primitives.append(hist.Clone(hist.GetName() + "_prim"))
                if 'linecolor'   not in kwargs: kwargs['linecolor']   = ROOT.kGray + 3
                if 'markercolor' not in kwargs: kwargs['markercolor'] = ROOT.kGray + 3
                self._style(hist, **kwargs)
                self._legends[-1].AddEntry(hist, name, kwargs.get('option', 'L'))
                pass
            pass

        self._legends[-1].Draw()

        # Clear entries (allowing for multiple legends)
        self._clear_all_entries()
        return



    # Private accessor methods
    # ----------------------------------------------------------------

    def _bare (self):
        """ ... """

        self._pad
        return


    def _xaxis (self):
        """ ... """

        # Return axis
        primitive = self._get_first_primitive()

        if primitive is None:
            warning("Cannot access x-axis")
            return None

        if not hasattr(primitive, 'GetXaxis'):
            return None

        return primitive.GetXaxis()


    def _yaxis (self):
        """ ... """

        # Return axis
        primitive = self._get_first_primitive()

        if primitive is None:
            warning("Cannot access y-axis")
            return None

        if not hasattr(primitive, 'GetYaxis'):
            return None

        return primitive.GetYaxis()


    def _get_first_primitive (self):
        """ ... """

        # Check(s)
        if len(self._pad.GetListOfPrimitives()) < 2:
            warning("Nothing was drawn; cannot access first primitive.")
            return None

        # Return first primitive
        return self._pad.GetListOfPrimitives()[1]



    # Private plotting methods
    # ----------------------------------------------------------------

    def _plot (self, plottype, data, display=True, **kwargs):
        """ ... """

        # Get plot option
        if 'option' not in kwargs:
            kwargs['option'] = self._get_plot_option(plottype)
            pass

        if type(data).__module__.startswith(np.__name__) or type(data) == list:
            # Numpy-/list-type
            if plottype == PlotType.stack:
                scale = kwargs.pop('scale', None) # Scale only once!
                hist = self._plot1D_numpy(data, display=False,   scale=scale, **kwargs)
                return self._plot1D_stack(hist, display=display, **kwargs)
            else:
                return self._plot1D_numpy(data, display=display, **kwargs)

        elif type(data).__name__.startswith('TH1') or type(data).__name__.startswith('TProfile') or type(data).__name__.startswith('TGraph'):
            # ROOT 1D-type
            if plottype == PlotType.stack:
                scale = kwargs.pop('scale', None) # Scale only once!
                hist = self._plot1D      (data, display=False,   scale=scale, **kwargs)
                return self._plot1D_stack(hist, display=display, **kwargs)
            else:
                hist = data.Clone(data.GetName() + '_clone')
                return self._plot1D      (hist, display=display, **kwargs)

        elif type(data).__name__.startswith('TH2'):
            # ROOT 2D-type
            assert plottype == PlotType.hist2d
            hist = data.Clone(data.GetName() + '_clone')
            return self._plot1D      (hist, display=display, **kwargs)  # @TODO: _plot2D?
            
        else:
            warning("_plot: Input data type not recognised:")
            print type(data[0])
            pass

        return None


    def _ratio_plot (self, plottype, data, **kwargs):
        """ ... """

        # Check(s)
        assert type(data[0]) == type(data[1]), "Input data types must match"

        # Get plot option
        if 'option' not in kwargs:
            kwargs['option'] = self._get_plot_option(plottype)
            pass

        if type(data[0]).__module__.startswith(np.__name__) or type(data) == list:
            # Numpy-/list-type
            return self._ratio_plot1D_numpy(data, **kwargs)

        elif type(data[0]).__name__.startswith('TH1') or type(data[0]) == ROOT.TProfile:
            # ROOT-type
            return self._ratio_plot1D      (data, **kwargs)

        else:
            warning("_ratio_plot: Input data type not recognised:")
            print type(data[0])
            pass
        return None


    def _diff_plot (self, plottype, data, **kwargs):
        """ ... """

        # Check(s)
        assert type(data[0]) == type(data[1]), "Input data types must match"

        # Get plot option
        if 'option' not in kwargs:
            kwargs['option'] = self._get_plot_option(plottype)
            pass

        if type(data[0]).__module__.startswith(np.__name__) or type(data) == list:
            # Numpy-/list-type
            return self._diff_plot1D_numpy(data, **kwargs)

        elif type(data[0]).__name__.startswith('TH1'):
            # ROOT TH1-type
            return self._diff_plot1D      (data, **kwargs)

        else:
            warning("_diff_plot: Input data type not recognised:")
            print type(data[0])
            pass
        return None


    def _plot1D_numpy (self, data, bins, weights=None, option='', **kwargs):
        """ ... """

        # Check(s)
        if bins is None:
            warning("You need to specify 'bins' when plotting a numpy-type input.")
            return

        if len(bins) < 2:
            warning("Number of bins {} is not accepted".format(len(bins)))
            return

        # Fill histogram
        if len(data) == len(bins):
            # Assuming 'data' and 'bins' are sets of (x,y)-points
            h = ROOT.TGraph(len(bins), np.array(bins), np.array(data))
        else:
            h = ROOT.TH1F('h_{:d}'.format(int(time.time()*1E+06)), "", len(bins) - 1, np.array(bins))
            if len(data) == len(bins) - 1:
                # Assuming 'data' are bin values
                array2hist(data, h)
            else:
                # Assuming 'data' are values to be filled
                fill_hist(h, data, weights=weights)
                pass
            pass

        # Plot histogram
        return self._plot1D(h, option, **kwargs)


    def _ratio_plot1D_numpy (self, data, bins, weights=None, option='', **kwargs):
        """ ... """

        # Check(s)
        if bins is None:
            warning("You need to specify 'bins' when plotting a numpy-type input.")
            return

        if len(bins) < 2:
            warning("Number of bins {} is not accepted".format(len(bins)))
            return

        # Fill histogram
        h1 = ROOT.TH1F('h_num_{}'.format(id(data)), "", len(bins) - 1, bins)
        h2 = ROOT.TH1F('h_den_{}'.format(id(data)), "", len(bins) - 1, bins)
        fill_hist(h1, data[0], weights=weights[0])
        fill_hist(h2, data[1], weights=weights[1])

        return _ratio_plot1D((h1,h2), option, **kwargs)


    def _diff_plot1D_numpy (self, data, bins, weights=None, option='', **kwargs):
        """ ... """

        # Check(s)
        if bins is None:
            warning("You need to specify 'bins' when plotting a numpy-type input.")
            return

        if len(bins) < 2:
            warning("Number of bins {} is not accepted".format(len(bins)))
            return

        # Fill histogram
        h1 = ROOT.TH1F('h_num_{}'.format(id(data)), "", len(bins) - 1, bins)
        h2 = ROOT.TH1F('h_den_{}'.format(id(data)), "", len(bins) - 1, bins)
        fill_hist(h1, data[0], weights=weights[0])
        fill_hist(h2, data[1], weights=weights[1])

        return _diff_plot1D((h1,h2), option, **kwargs)


    @cd
    @update
    def _plot1D (self, hist, option='', display=True, scale=None, **kwargs):
        """ ... """

        # Check(s)
        # ...

        # Normalise
        if 'normalise' in kwargs and kwargs['normalise']:
            if hist.Integral() > 0.:
                hist.Scale(1./hist.Integral())
                #hist.Scale(1./hist.Integral(0, hist.GetXaxis().GetNbins() + 1))
                pass
            pass

        # Scale
        if scale is not None and type(hist) != ROOT.THStack:
            hist.Scale(scale)
            pass

        # Style
        self._reset_style(hist, option)
        self._style(hist, **kwargs)

        # Append draw option (opt.)
        if is_overlay(self):
            option += " ][ SAME"
        elif len(self._primitives) > 0:
            option += " SAME"
            pass

        # Only plot if requested
        if display:

            # Draw histograms
            if type(hist) in [ROOT.THStack, ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]:
                hist.Draw(option)
            else:
                hist.DrawCopy(option)
                pass

            # Store reference to primitive
            self._add_to_primitives(hist)
            hist = self._primitives[-1] # Reference the stored histogram

            # Check whether several filled histograms have been added
            if (type(hist) == ROOT.THStack or hist.GetFillColor() != 0) and len(filter(lambda h: type(h) == ROOT.THStack or (type(h).__name__.startswith('TH') and h.GetFillColor() != 0 and not option.startswith('E')), self._primitives)) == 2:
                warning("Several filled, non-stacked histograms have been added. This may be misleading.")
                pass

            if type(hist) != ROOT.THStack:
                # Store legend entry
                if 'label' in kwargs and kwargs['label'] is not None:

                    opt = kwargs.get('legend_option', self._get_label_option(option, hist))

                    if 'data' in kwargs['label'].strip().lower():
                        self._entries.insert(0, (hist, kwargs['label'], opt))
                    else:
                        self._entries.append((hist, kwargs['label'], opt))
                        pass
                    pass

                pass

            # Out-of-bounds markers
            if 'oob' in kwargs and kwargs['oob'] and 'oob' not in hist.GetName():
                if self._oob_up or self._oob_down:
                    warning("Out-of-bounds markers already exists.")
                    pass
                if self._ylim is None:
                    warning("Y-axis limits not set.")
                    pass
                self._oob_up   = hist.Clone(hist.GetName() + '_oob_up')
                self._oob_down = hist.Clone(hist.GetName() + '_oob_down')
                ymin, ymax = self.ylim()
                diff = ymax - ymin
                for bin in range(1, hist.GetXaxis().GetNbins() + 1):
                    c = hist.GetBinContent(bin)
                    if c > ymax: self._oob_up  .SetBinContent(bin, ymin + diff * 0.9)
                    else:        self._oob_up  .SetBinContent(bin, -9999.)
                    if c < ymin: self._oob_down.SetBinContent(bin, ymin + diff * 0.1)
                    else:        self._oob_down.SetBinContent(bin, -9999.)
                    self._oob_up  .SetBinError(bin, 0)
                    self._oob_down.SetBinError(bin, 0)
                    pass
                self._plot1D(self._oob_up,   markercolor=ROOT.kBlue, markerstyle=22, markersize=1.2, option='P HIST')
                self._plot1D(self._oob_down, markercolor=ROOT.kBlue, markerstyle=23, markersize=1.2, option='P HIST')
                # ...
                pass

            pass

        return hist # .Clone(hist.GetName().replace('_clone', ''))


    def _ratio_plot1D (self, hists, option='', offset=None, default=1, **kwargs):
        """ ... """

        # Check(s)
        if type(hists[0]) == ROOT.TProfile:
            # Create a new TH1 histogram, instead of cloning, in case inputs are TProfiles for which SetBinContent makes little sense.
            ax = hists[0].GetXaxis()
            h = ROOT.TH1F(hists[0].GetName() + '_ratio', "", ax.GetNbins(), ax.GetXmin(), ax.GetXmax())
        else:
            # Clone if inputs are standard ROOT TH1*'s , in order to keep any style applied previously
            h = hists[0].Clone(hists[0].GetName() + '_ratio')
            pass

        # Fill bins with ratio
        for bin in range(1, h.GetXaxis().GetNbins() + 1):
            num   = hists[0].GetBinContent(bin)
            num_e = hists[0].GetBinError  (bin)
            denom = hists[1].GetBinContent(bin)
            h.SetBinContent(bin, num   / denom if denom > 0 else default)
            h.SetBinError  (bin, num_e / denom if denom > 0 else 9999.)
            pass


        # Add offset (opt.)
        if offset is not None:
            h_offset = h.Clone(h.GetName() + '_offset')
            for bin in range(1, h.GetXaxis().GetNbins() + 1):
                h       .SetBinContent(bin, offset + h.GetBinContent(bin))
                h_offset.SetBinContent(bin, offset)
                h_offset.SetBinError  (bin, 0)
                pass
            pass

        # Plot histogram
        result = self._plot1D(h, option, **kwargs)
        if offset is not None:
            self._plot1D(h_offset, 'HIST', fillcolor=10)
            self._get_first_primitive().Draw('AXIS SAME')
            pass
        return result


    def _diff_plot1D (self, hists, option='', offset=None, uncertainties=True, **kwargs):
        """ ... """

        h = hists[0].Clone(hists[0].GetName() + '_diff')
        if uncertainties:
            h.Add(hists[1], -1)
        else:
            for bin in range(1, h.GetXaxis().GetNbins() + 1):
                denom = hists[1].GetBinContent(bin)
                h.SetBinContent(bin, h.GetBinContent(bin) - denom)
                pass
            pass

        # Add offset (opt.)
        if offset is not None:
            h_offset = h.Clone(h.GetName() + '_offset')
            for bin in range(1, h.GetXaxis().GetNbins() + 1):
                h       .SetBinContent(bin, offset + h.GetBinContent(bin))
                h_offset.SetBinContent(bin, offset)
                h_offset.SetBinError  (bin, 0)
                pass
            pass

        # Plot histogram
        result = self._plot1D(h, option, **kwargs)
        if offset is not None:
            self._plot1D(h_offset, 'HIST', fillcolor=10)
            self._get_first_primitive().Draw('AXIS SAME')
            pass
        return result


    @update
    def _plot1D_stack (self, hist, option='', **kwargs):
        """ ... """

        # Manually add to legend entries
        if 'label' in kwargs:
            # Add in the correct (inverse) order for stacked histograms
            idx = 1 if (len(self._entries) > 0 and self._entries[0][1].strip().lower() == 'data') else 0
            self._entries.insert(idx, (hist, kwargs['label'], self._get_label_option(option, hist)))
            pass

        # Scale etc.
        kwargs['display'] = False
        hist = self._plot1D(hist, option=option, **kwargs)
        kwargs.pop('display')

        first = self._add_to_stack(hist)
        if first:
            self._plot1D(self._stack, option=option, **kwargs)
            pass

        return hist


    def _add_to_stack (self, hist, option='HIST'):
        """ ... """

        first = False
        if self._stack is None:
            self._stack = ROOT.THStack('stack', "")
            first = True
            pass

        self._stack.Add(hist.Clone(hist.GetName() + "_stack"), option)
        return first


    def _add_to_primitives (self, hist):
        """ ... """

        if type(hist).__name__.startswith('TH1'):
            self._primitives.append(hist)#.Clone(hist.GetName() + "_prim"))
        else:
            self._primitives.append(hist)
            pass
        return



    # Private cosmetics methods
    # ----------------------------------------------------------------

    def _bare (self):
        """ ... """
        return self._pad


    def _get_all_entries (self):
        """ ... """
        return self._entries + [entry for child in self._children for entry in child._entries]

    def _clear_all_entries (self):
        """ ... """
        self._entries = list()
        for child in self._children:
            child._clear_all_entries()
            pass
        return


    @update
    def _update (self, only_this=True):
        """ ...
        @TODO: - This is some reaaaally shitty naming convention
        """

        # Check(s)
        if len(self._primitives) == 0 or not hasattr(self._pad, 'SetLogy'): return

        # Set x-axis limits
        if self._xlim:
            self._xaxis().SetRangeUser(*self._xlim)
            pass

        # Set y-axis log./lin. scale
        self._pad.SetLogy(self._log)
        self._pad.SetLogx(self._logx)

        # Set y-axis limits with padding
        axisrange = (None,None)
        if self._ylim:
            axisrange = self._ylim
        else:
            ymin, ymax = inf, -inf

            try:
                ymin = min(filter(lambda y: y is not None, map(get_minimum, self._primitives)))
            except ValueError: # only stacked histogram
                ymin = 0.
                pass

            #ymin_positive = 100. #
            ymax = max(map(get_maximum, self._primitives))
            #for hist in self._primitives:
            #    ymax = max(get_maximum(hist), ymax)
            #    pass
            if ymax is None:
                print "WARNING: Got `ymax == None`. Unable to set axis ranges properly."
            else:
                if self._log:
                    if self._ymin:
                        ymin_positive = self._ymin
                    else:
                        ymin_positive = min(filter(lambda y: y is not None, map(get_minimum_positive, self._primitives)))
                        ymin_positive *= 0.8
                        pass
                    axisrange = (ymin_positive, np.exp((np.log(ymax) - np.log(ymin_positive)) / (1. - self._padding) + np.log(ymin_positive)))
                else:
                    axisrange = (0, ymax / (1. - self._padding) )
                    pass

                # Set overlay axis limits
                if is_overlay(self):
                    # self.lim(ymin, ymax)
                    self.lim(0, ymax, force=False) # @TODO: Fix. Getting ymin == ymax
                    pass
                pass
            pass

        # Check if anything has been drawn
        if self._get_first_primitive() and self._yaxis():
            self._get_first_primitive().SetMinimum(axisrange[0]) # For THStack. @TODO: Improve?
            self._get_first_primitive().SetMaximum(axisrange[1]) # ...
            self._yaxis().SetRangeUser(*axisrange)
            if hasattr(self._get_first_primitive(), 'GetHistogram'):
                self._get_first_primitive().GetHistogram().SetMinimum(axisrange[0])
                self._get_first_primitive().GetHistogram().SetMaximum(axisrange[1])
                self._get_first_primitive().GetHistogram().GetYaxis().SetRangeUser(*axisrange)
                pass

            # Style
            # @TODO: Move into a 'style' method
            if is_canvas(self._base): # if 'pad' on 'canvas'
                self._yaxis().SetTitleOffset(ROOT.gStyle.GetTitleOffset('y') * self._base._size[1]       / float(self._base._size[0]))
                pass

            self._xaxis().SetTickLength(ROOT.gStyle.GetTickLength('x') * self._scale[1])
            self._yaxis().SetTickLength(ROOT.gStyle.GetTickLength('y') * self._scale[0])
            pass

        # Perform overlay pad-specific update
        if is_overlay(self):
            self._update_overlay()
            pass

        return


    def _style (self, h, **kwargs): # @TODO: Should these be utility functions?
        """ ..."""

        # Check(s)
        if type(h) == ROOT.THStack: return

        # Dispatch style methods
        dispatch = {
            'fillstyle': h.SetFillStyle,
            'fillcolor': h.SetFillColor,

            'linestyle': h.SetLineStyle,
            'linecolor': h.SetLineColor,
            'linewidth': h.SetLineWidth,

            'markerstyle': h.SetMarkerStyle,
            'markercolor': h.SetMarkerColor,
            'markersize':  h.SetMarkerSize,
        }

        for var, setter in dispatch.items():
            if var in kwargs:
                setter(kwargs[var])
                pass
            pass

        if 'alpha' in kwargs:
            if 'fillcolor' in kwargs:
                h.SetFillColorAlpha(kwargs['fillcolor'], kwargs['alpha'])
            else:
                warning("Set 'alpha' without 'fillcolor'.")
                pass
            pass

        return


    def _style_line (self, l, **kwargs): # @TODO: Should these be utility functions?
        """ ..."""

        # Check(s)
        # ...

        # Dispatch style methods
        dispatch = {
            'linestyle': l.SetLineStyle,
            'linecolor': l.SetLineColor,
            'linewidth': l.SetLineWidth,
        }

        for var, setter in dispatch.items():
            if var in kwargs:
                setter(kwargs[var])
                pass
            pass

        return


    def _style_text (self, t, **kwargs): # @TODO: Should these be utility functions?
        """ ..."""

        # Check(s)
        # ...

        # Dispatch style methods
        dispatch = {
            'textfont':  t.SetTextFont,
            'textcolor': t.SetTextColor,
            'textsize':  t.SetTextSize,
            'textalign': t.SetTextAlign,
            'textangle': t.SetTextAngle,
        }

        for var, setter in dispatch.items():
            if var in kwargs:
                setter(kwargs[var])
                pass
            pass

        return


    def _reset_style (self, h, option): # @TODO: Should these be utility functions?
        """ ... """

        # Check(s)
        if type(h) == ROOT.THStack: return

        option = option.strip().split()[0].upper()
        if 'P' not in option:
            h.SetMarkerSize (0)
            h.SetMarkerStyle(0)
            h.SetMarkerColor(0)
            pass

        # ...
        return


    def _get_plot_option (self, plottype):
        """ ... """

        option = ''
        if   plottype == PlotType.plot:  option = 'PE0'
        elif plottype == PlotType.hist:  option = 'HIST'
        elif plottype == PlotType.stack: option = 'HIST'
        elif plottype == PlotType.graph: option = ('A' if len(self._primitives) == 0 else '') + 'PE0' # 'PL'
        else:
            warning("Plot type '{}' not recognised".format(plottype.name))
            pass

        return option


    def _get_label_option (self, plot_option, hist):
        """ ... """

        label_option = ''

        plot_option = plot_option.split()[0].upper() # Discard possible 'SAME'

        if 'L' in plot_option:
            label_option += 'L'
        elif 'HIST' in plot_option:
            if hist.GetFillColor() == 0:
                label_option += 'L'
            else:
                label_option += 'F'
                pass
            pass

        if 'P' in plot_option:
            label_option += 'P'
            pass

        if 'E2' in plot_option or 'E3' in plot_option:
            label_option += 'F'
        elif 'E' in plot_option:
            label_option += 'EL'
            pass

        return label_option

    pass
