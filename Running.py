import numpy as np
from numpy import sqrt
from numpy import cosh
from numpy import tanh
import scipy
import matplotlib.pyplot as plt
from numpy import pi, log, exp, sqrt
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d, splev, splrep
from scipy.optimize import fsolve
import scipy.optimize as opt
import cmath
import math


def Csch(z):
    return 1/np.sinh(z)
def Sech(z):
    return 1/np.cosh(z)


# Let's work in Planck units
Mp = 1

class Running_V:

    def __init__(self, 
                 Inflation_template=None,  # template model to be chosen among: None, "T-model" , "E-model", "Powerlaw"
                 feature_template=None,    # template feature shape to be chosen among: None, "Gaussian-Bump"
                 Vi   = None, 
                 dVi  = None,     # User-defined potential and its derivatives
                 d2Vi = None,     # arguments need to take the form (phi, [p1, p2, ...])
                 d3Vi = None,     # where p1, p2, ... are the parameters entering the potential
                 d4Vi = None,     
                 Ninf = 55,       # Number of e-folds after horizon exit, until the end of inflation
                 debug = False):
       


        if(debug):
            print('##########################')
            print('#     Initialisation     #')
            print('##########################')

        self.Inflation_template     = Inflation_template         # template model to be chosen among: None, "T-model" , "E-model", "Powerlaw"
        
        self.feature_template       = feature_template           # template feature shape to be chosen among: None, "Gaussian-Bump"
        
        
        self.Ninf = Ninf      # Number of e-folds after horizon exit, until the end of inflation

        self.debug = debug        # do we want to plot and print stuff?

        # to be set to False in case of any problem
        self.successful_initialization = True

        ####################################
        #
        #   Model definition
        #
        ####################################
        
        if(debug):
            print('+ Chosing potential...')
        
        #########################
        # user-defined potential
        #########################
        if(Inflation_template==None):  #feature_template==None an feature_template_param==None
            
            if(debug): 
                print("+ Potential is user-defined")

            def Veff(phi, params):   return Vi(phi, params)
            def dVeff(phi, params):  return dVi(phi, params)
            def d2Veff(phi, params): return d2Vi(phi, params)
            def d3Veff(phi, params): return d3Vi(phi, params)
            def d4Veff(phi, params): return d4Vi(phi, params)
        
        #########################
        # T-models
        #########################
        elif(Inflation_template=="T-model"):

            if(debug):
                print('+ Model chosen: T-model')

            def Veff(phi, params):
                V0, alpha, n = params
                return V0*tanh(phi/(sqrt(6)*sqrt(alpha)*Mp))**(2*n)

            def dVeff(phi, params):
                V0, alpha, n = params
                return (sqrt(0.6666666666666666)*n*V0*Sech(phi/(sqrt(6)*sqrt(alpha)*Mp))**2*tanh(phi/(sqrt(6)*sqrt(alpha)*Mp))**(-1 + 2*n))/(sqrt(alpha)*Mp)
            
            def d2Veff(phi, params):
                V0, alpha, n = params
                return (-4*n*V0*(-2*n + cosh((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))*Csch((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))**2*tanh(phi/(sqrt(6)*sqrt(alpha)*Mp))**(2*n))/(3.*alpha*Mp**2)

            def d3Veff(phi, params):
                V0, alpha, n = params
                return (2*sqrt(0.6666666666666666)*n*V0*(3 + 8*n**2 - 12*n*cosh((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)) + cosh((2*sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))*Csch((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))**3*tanh(phi/(sqrt(6)*sqrt(alpha)*Mp))**(2*n))/(3.*alpha**1.5*Mp**3)

            def d4Veff(phi, params):
                V0, alpha, n = params
                return (-2*n*V0*(-60*n - 32*n**3 + (23 + 96*n**2)*cosh((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)) - 28*n*cosh((2*sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)) + cosh((sqrt(6)*phi)/(sqrt(alpha)*Mp)))*Csch((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))**4*tanh(phi/(sqrt(6)*sqrt(alpha)*Mp))**(2*n))/(9.*alpha**2*Mp**4)

            if(debug): print('+ Potential and its derivatives are stored.')

        #########################
        # E-models
        #########################
        elif(Inflation_template=="E-model"):
            
            if(debug):
                print('+ Model chosen: E-model') 

            def Veff(phi, params):
                V0, alpha, n = params
                return (1 - np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))**(2*n)*V0

            def dVeff(phi, params):
                V0, alpha, n = params
                return (-2*sqrt(0.6666666666666666)*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(1 - np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))**(-1 + 2*n)*n*V0)/(sqrt(alpha)*Mp)

            def d2Veff(phi, params):
                V0, alpha, n = params
                return (4*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(1 - np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))**(2*(-1 + n))*n*(-1 + 2*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*n)*V0)/(3.*alpha*Mp**2)

            def d3Veff(phi, params):
                V0, alpha, n = params
                return (-4*sqrt(0.6666666666666666)*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(1 - np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))**(-3 + 2*n)*n*(1 + np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(1 - 6*n) + 4*np.exp((2*sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*n**2)*V0)/(3.*alpha**1.5*Mp**3)

            def d4Veff(phi, params):
                V0, alpha, n = params
                return (8*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(1 - np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp)))**(2*(-2 + n))*n*(-1 + 8*np.exp((sqrt(6)*phi)/(sqrt(alpha)*Mp))*n**3 + 2*np.exp((sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(-2 + 7*n) + np.exp((2*sqrt(0.6666666666666666)*phi)/(sqrt(alpha)*Mp))*(-1 + 8*n - 24*n**2))*V0)/(9.*alpha**2*Mp**4)

            if(debug): print('+ Potential and its derivatives are stored.')

        #########################
        # powerlaw: alpha is the power, n is note used
        #########################
        elif(Inflation_template=="Powerlaw"):
            
            if(debug):
                print('+ Model chosen: Powerlaw')

            def Veff(phi, params):
                V0, alpha = params
                return (np.abs(phi)/Mp)**(alpha)*V0

            def dVeff(phi, params):
                V0, alpha = params
                return alpha*(np.sign(phi)/Mp)*(np.abs(phi)/Mp)**(alpha-1)*V0
            
            def d2Veff(phi, params):
                V0, alpha = params
                return alpha*(alpha-1)*(np.sign(phi)/Mp)**2*(np.abs(phi)/Mp)**(alpha-2)*V0
            
            def d3Veff(phi, params):
                V0, alpha = params
                return alpha*(alpha-1)*(alpha-2)*(np.sign(phi)/Mp)**3*(np.abs(phi)/Mp)**(alpha-3)*V0
            
            def d4Veff(phi, params):
                V0, alpha = params
                return alpha*(alpha-1)*(alpha-2)*(alpha-3)*(np.sign(phi)/Mp)**4*(np.abs(phi)/Mp)**(alpha-4)*V0

            if(debug): print('+ Potential and its derivatives are stored.')
        else:
            self.successful_initialization = False
            if(debug):
                print('==> ERROR: Model is ill-defined. Either not all derivatives are provided, '
                'or template and its corresponding parameters are not consistently provided.')
                print('Aborting.')
        

        ###################################
        #  Transfer function definition
        ###################################
        
        # no feature
        if(feature_template==None):

            def T(phi, params=None):
                return 1
        
            def dT(phi, params=None):
                return 0
            
            def d2T(phi, params=None):
                return 0
            
            def d3T(phi, params=None):
                return 0
            
            def d4T(phi, params=None):
                return 0

        # Gaussian bump
        elif(feature_template=="Gaussian-Bump"):
            
            def T(phi, params):
                loc, ampli, width = params
                return 1 + ampli/(sqrt(2)*np.exp((loc - phi)**2/(2.*width**2))*sqrt(np.pi)*width)
        
            def dT(phi, params):
                loc, ampli, width = params
                return (ampli*(loc - phi))/(sqrt(2)*np.exp((loc - phi)**2/(2.*width**2))*sqrt(np.pi)*width**3)
            
            def d2T(phi, params):
                loc, ampli, width = params
                return (ampli*(loc**2 - 2*loc*phi + phi**2 - width**2))/(sqrt(2)*np.exp((loc - phi)**2/(2.*width**2))*sqrt(np.pi)*width**5)
            
            def d3T(phi, params):
                loc, ampli, width = params
                return (ampli*(loc - phi)*(loc**2 - 2*loc*phi + phi**2 - 3*width**2))/(sqrt(2)*np.exp((loc - phi)**2/(2.*width**2))*sqrt(np.pi)*width**7)
            
            def d4T(phi, params):
                loc, ampli, width = params
                return (ampli*(loc**4 - 4*loc**3*phi + phi**4 - 6*phi**2*width**2 + 3*width**4 + 6*loc**2*(phi**2 - width**2) - 4*loc*(phi**3 - 3*phi*width**2)))/(sqrt(2)*np.exp((loc - phi)**2/(2.*width**2))*sqrt(np.pi)*width**9)
        
        else:
            self.successful_initialization=False
            if(debug):
                print('==> ERROR: transfer function type is unknown. Aborting.')

        if(self.successful_initialization):

            ############## Potential ################
            def V(phi, params_V, params_T):
                return T(phi, params_T)*Veff(phi, params_V)

            def dV(phi, params_V, params_T):
                return dVeff(phi, params_V)*T(phi, params_T) + dT(phi, params_T)*Veff(phi, params_V)

            def d2V(phi, params_V, params_T):
                return 2*dT(phi, params_T)*dVeff(phi, params_V) + d2Veff(phi, params_V)*T(phi, params_T) + d2T(phi, params_T)*Veff(phi, params_V)

            def d3V(phi, params_V, params_T):
                return 3*d2Veff(phi, params_V)*dT(phi, params_T) + 3*d2T(phi, params_T)*dVeff(phi, params_V) + d3Veff(phi, params_V)*T(phi, params_T) + d3T(phi, params_T)*Veff(phi, params_V)

            def d4V(phi, params_V, params_T):
                return 6*d2T(phi, params_T)*d2Veff(phi, params_V) + 4*d3Veff(phi, params_V)*dT(phi, params_T) + 4*d3T(phi, params_T)*dVeff(phi, params_V) + d4Veff(phi, params_V)*T(phi, params_T) + d4T(phi, params_T)*Veff(phi, params_V)

            self.V = V
            self.dV = dV
            self.d2V = d2V
            self.d3V = d3V
            self.d4V = d4V


        if(debug):
            if(self.successful_initialization):             
                print('----> Initialization was succesfull.')
            else:
                print('----> Initialization Failed.')
        
        
    ###################################################################
    ##
    ##  Find the PS running parameters (A_s, n_s, alpha_s, beta_s)
    ##
    ###################################################################
    
    def find_running(self, params_V=None, params_T=None):
        
        debug = self.debug
        Ninf = self.Ninf

        def V(phi):   return self.V(phi, params_V, params_T)
        def dV(phi):  return self.dV(phi, params_V, params_T)
        def d2V(phi): return self.d2V(phi, params_V, params_T)
        def d3V(phi): return self.d3V(phi, params_V, params_T)
        def d4V(phi): return self.d4V(phi, params_V, params_T)

        # if initialisation was successful, proceed.
        if(self.successful_initialization):

            if(debug):
                print('##########################')
                print('#    Finding Phi_star    #')
                print('##########################')

            #Let's start at a random field value at rest            
            Phistart = -1
            dPhistart = 0
            
            # give it a large number of e-folds to find the end of inflation
            Nmax = 10000



            ####################################################
            ############## Slow Roll Parameters ################
            ####################################################
            def H2(phi, dphi, V):
                return (2*V(phi))/((6*Mp**2) - dphi**2)
            
            def epsilonH(dphi):
                return dphi**2 / (2*Mp**2)

            ############## Solver EoM ################
            def dy(t, y, V, dV):
                y0, y1 = y[0], y[1]
                dy0 = y1
                dy1 = (epsilonH(y1)-3)*y1 - dV(y0)/H2(y0,y1,V)
                return [dy0, dy1]

            ############## Stopping conditions ################
            
            # reaching the end of inflation
            def endInflat(t, y):
                return y[1]**2 / (2*Mp**2) -1                                       
            endInflat.terminal = True
            endInflat.direction = +1

            # turning back before inflation ended
            def stuck(t, y):
                return y[1]                                       
            stuck.terminal = True
            stuck.direction = -1

            ############## Solve until we find a full inflation story ################
            passed = 0

            while(passed==0):
                y_init = [Phistart,dPhistart]
                sol = solve_ivp(lambda t,y : dy(t,y,V,dV), [0,Nmax], y_init, events=[endInflat, stuck], rtol=1e-12 , atol=1e-12)
                if(sol.t[-1]-sol.t[0]<65):
                    Phistart = 1.5*Phistart
                elif(np.abs((sol.t[-1]-Nmax))<1):
                    Phistart = 0.8*Phistart
                else:
                    passed = 1

            if(debug): print("+ Solving EoM: Passed")
            
            ############## Create tables and interpolations ################

            listPhi = sol.y[0]
            listdPhi = sol.y[1]
            listEpsilon = np.array([epsilonH(dPhi) for dPhi in listdPhi])
            listlogeps=np.log(listEpsilon[np.where(listEpsilon>0)])
            logeps = splrep(sol.t[np.where(listEpsilon>0)],listlogeps, s=0)

            def dlogeps(N): return splev(N, logeps, der=1)
            
            def etaH(dphi, N):
                if dphi == 0:
                    return 0
                else :    
                    return epsilonH(dphi)-0.5*dlogeps(N)
                
            listEta = np.array([etaH(listdPhi[i],sol.t[i]) for i in range(len(sol.t))])
            
            NEnd = sol.t[-1]
            NStar = NEnd - Ninf

            Phi_interp = interp1d(sol.t,sol.y[0])
            dPhi_interp = interp1d(sol.t,sol.y[1])
            epsilon_interp = interp1d(sol.t,listEpsilon)
            eta_interp = interp1d(sol.t,listEta)
            # H2_interp = interp1d(sol.t,listH2)

            PhiStar = Phi_interp(NStar)
            dPhiStar = dPhi_interp(NStar)
            epsilonStar = epsilon_interp(NStar)
            etaStar = eta_interp(NStar)

            
            A = H2(PhiStar, dPhiStar, V)/(epsilonStar*8*np.pi**2)

            ############## Potential and its derivatives at horizon exit ################
            Vstar = V(PhiStar)
            dVstar = dV(PhiStar)
            d2Vstar = d2V(PhiStar)
            d3Vstar = d3V(PhiStar)
            d4Vstar = d4V(PhiStar)

            # from 1303.3787
            eps_1 = Mp**2/2 * (dVstar/Vstar)**2
            eps_2 = 2 * Mp**2 * ( (dVstar/Vstar)**2 - d2Vstar/Vstar )
            eps_3 = (1/eps_2) * 2 * Mp**4 *(d3Vstar*dVstar/(Vstar**2) - 3*d2Vstar/Vstar*(dVstar/Vstar)**2 + 2*(dVstar/Vstar)**4 )

            # calculated using notebook 'Potentials.nb'
            eps_4 =  (1/eps_3) * Mp**4 * dVstar * (4*dVstar**7 - 13*Vstar*dVstar**5*d2Vstar + Vstar**2*dVstar**4*d3Vstar + Vstar**4*d2Vstar**2*d3Vstar 
                                                + Vstar**2*dVstar**3*(14*d2Vstar**2 - Vstar*d4Vstar) + Vstar**3*dVstar*(-6*d2Vstar**3 - Vstar*d3Vstar**2 + Vstar*d2Vstar*d4Vstar))
            
            
            if(debug):

                print('eps_1 = ', eps_1)
                print('eps_2 = ', eps_2)
                print('eps_3 = ', eps_3)
                print('eps_4 = ', eps_4)

                print('ns=',1-6*epsilonStar+2*etaStar, ' and r=', 16*epsilonStar)
                print('Total number of efolds: ', NEnd-NStar)

                print('As =',A)

            ##### figuring out the running ######

            r = 16*epsilonStar
            n_s = 1 - 2*eps_1 - eps_2

            # from 2205.12608 (Appendix C.1, setting the delta's to zero since they are there for K-flation)
            alpha_s = - 2*eps_1*eps_2 - eps_2*eps_3
            beta_s = - 2*eps_1*eps_2**2 - 2*eps_1*eps_2*eps_3 - eps_2*eps_3**2 - eps_2*eps_3*eps_4

            if(debug):
                print('r = ',r)
                print('n_s = ',n_s)
                print('alpha_s = ',alpha_s)
                print('beta_s = ',beta_s)
            
            
            return [A, r, n_s, alpha_s, beta_s]

        # if initialization had failed, return zeros.
        else:
            return [0, 0, 0, 0, 0]
