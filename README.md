# Inflation_Running
Code aiming at constraining theoretical inflation potentials with available cosmological data

# Instructions

The Class "Running_V" needs to be called with the following arguments:

Inflation_template=None,  # template model to be chosen among: None, "T-model", "E-model", "Powerlaw"
feature_template=None,    # template feature shape to be chosen among: None, "Gaussian-Bump"
Vi   = None, 
dVi  = None,     # User-defined potential and its derivatives defined externally
d2Vi = None,     # arguments need to take the form (phi, [p1, p2, ...])
d3Vi = None,     # where p1, p2, ... are the parameters entering the potential
d4Vi = None,     
Ninf = 55,       # Number of e-folds after horizon exit, until the end of inflation
debug = False    # turn to True if debugging instructions are needed

The function "Running_V.find_running(self, params_V=None, params_T=None)" needs to be provided with tables of parameters both for the potential and the feature used (unless no feature is specified).
