# -*- coding: utf-8 -*-

""" Class derived from 'pad', allowing for overlaying pads on other pads

@file:   overlay.py
@date:   26 April 2017
@author: Andreas Søgaard 
@email:  andreas.sogaard@cern.ch
"""

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
   
from pad import pad


# Class definition
class overlay (pad):
    """
    docstring for overlay
    @TODO: Elaborate!
    """

    def __init__ (self, base, color=ROOT.kRed):
        """ Constructor. """
        # Check if canvas
        if hasattr(base, 'pads'):
            base = base.pads()[0]
            pass
        super(overlay, self).__init__(base, (0, 0, 1, 1)) # 'pad' contructor

        # Check(s)
        # ...
        
        base._update()
        
        # Add 'self' to canvas' list of pads
        idx = base._base._pads.index(base)
        base._base._pads.insert(idx + 1, self)
        

        # Member variables
        # -- Management
        self._base = base
        self._base_yaxis = base._yaxis()
        base._children.append(self)
        
        # -- Plotting cosmetics
        self._axis  = None
        self._xmin  = 0
        self._xmax  = 0
        self._ymin  = 0
        self._ymax  = 1
        self._lims_set = False
        self._color = color
        self._label = None

        # Resize canvas and pad(s)
        right_margin = 0.12
        c = base._base._bare() # Getting parent TCanvas; assuming 'canvas' > 'pad' > 'overlay' structure. @TODO: Improve?
        w_initial = 1 - c.GetLeftMargin() - c.GetRightMargin()
        w_final   = 1 - c.GetLeftMargin() - right_margin           
        c.SetCanvasSize(int(c.GetWw() * w_initial / w_final), c.GetWh())
        for p in base._base._pads:
            p._bare().SetRightMargin(right_margin)
            pass

        base._bare().SetTicks(1,0) # Remove y-axis tick on right-hand side
        base._bare().Update()
        base._bare().cd()


        # Store coordinates
        self._xmin = base._xaxis().GetXmin()
        self._xmax = base._xaxis().GetXmax()
        
        # Draw overlay pad and axis
        self._pad.Draw()
        self._pad.cd()

        # Axis
        self._update_axis()

        self._pad.Modified()
        self._pad.Update()
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



    # Public accessor/mutator methods
    # ----------------------------------------------------------------

    def lim (self, ymin, ymax, force=True):
        """ ... """
        
        # Check(s)
        assert ymin < ymax, "Axis limits must be given in increasing order; recieved (%.1e, %.1e)" % (ymin, ymax)

        # Decide whether to set limits
        if force or not self._lims_set:
            self._ymin = ymin
            self._ymax = ymax
            self._lims_set = force
            self._update_axis()
            pass
        
        return


    @update
    def label (self, label):
        """ ... """
        self._label = label
        return


    @update
    def ylabel (self, label):
        """ ... """
        return self.label(label)



    # Private plotting methods
    # ----------------------------------------------------------------

    @update
    def _update_overlay (self):
        """ ... """
        
        # Pad
        if self._pad:
            self._pad.SetFillStyle(4000)
            self._pad.SetFrameFillStyle(4000)
            self._pad.SetFrameFillColor(0)
            pass

        # Axis
        if self._axis:
            self._update_axis()
            self._axis.SetLabelFont(ROOT.gStyle.GetTextFont())
            self._axis.SetLabelSize(ROOT.gStyle.GetTextSize())
            self._axis.SetTitleFont(ROOT.gStyle.GetTextFont())
            self._axis.SetTitleSize(ROOT.gStyle.GetTextSize())
            self._axis.SetLineColor (self._color)
            self._axis.SetLabelColor(self._color)
            self._axis.SetTitleColor(self._color)
            self._axis.SetLabelOffset(ROOT.gStyle.GetLabelOffset('y'))
            if is_canvas(self._base._base):
                self._axis.SetTitleOffset(ROOT.gStyle.GetTitleOffset('z') * self._base._base._size[1] / float(self._base._base._size[0]))
                pass
            pass

        return


    @cd
    @update
    def _update_axis (self):
        """ ... """

        # Getting parent TCanvas; assuming 'canvas' > 'pad' > 'overlay' structure. @TODO: Improve?
        base = self._base._bare()
        
        # Set proper ranges
        # @TODO: Make prettier?
        if self._lims_set:
            # Limits set manually; respect them
            ymin = self._ymin
            ymax = self._ymax
        else:
            # Limits inferred from pad's primitives; add padding
            ymin = self._ymin
            if self._log:
                ymax = np.exp((np.log(self._ymax) - np.log(self._ymin)) / (1. - self._padding) + np.log(self._ymin))
            else:
                ymax = self._ymax / (1. - self._padding)
                pass
            pass

        dy   = (ymax       - ymin)       / (1. - base.GetTopMargin()  - base.GetBottomMargin())
        dx   = (self._xmax - self._xmin) / (1. - base.GetLeftMargin() - base.GetRightMargin())
      
        # (Re-) set pad ranges
        self._pad.Range(self._xmin - base.GetLeftMargin()   * dx,
                        ymin       - base.GetBottomMargin() * dy,
                        self._xmax + base.GetRightMargin()  * dx,
                        ymax       + base.GetTopMargin()    * dy)

        # Create and draw axis
        self._axis = ROOT.TGaxis(self._xmax, ymin,
                                 self._xmax, ymax, 
                                 ymin, ymax, 510, "+L")
        self._axis.Draw()

        # Set axis label (opt.)
        if self._label:
            self._axis.SetTitle(self._label)
            pass

        return

    pass
    
