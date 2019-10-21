import numpy as np
from scipy import interpolate, linalg, optimize
from optparse import OptionParser
from collections import OrderedDict
import pickle
import time

usage = 'usage: %prog [options]'
parser = OptionParser(usage)
parser.add_option("-d", "--details", action="store", type="int", default=5, dest="DET", help="1: Do pheno Q fit with JLA only. 2: Fit for a non scale dependent dipolar modulation in Q. 3: Fit for a top hat scale dependent dipolar modulation in Q. 4. Fit for an exponentially falling scale dependent dipolar modulation in Q. 5. Fit for a linearly falling scale dependent dipolar modulation in Q. ")
parser.add_option("-v", "--verbose", action = "store_true", default=False, dest="VERB", help = "Want lots of diagnostic outputs?")
parser.add_option("-p", "--pecvelcov", action = "store_true", default=True, dest="PVCO", help = "Exclude the peculiar velocity covariance matrix?")
parser.add_option("-f", "--forcezcmb", action = "store_true", default=False, dest="FZCMB", help = "Use Zcmb instead of zhel")
parser.add_option("-r", "--reversebias", action = "store_true", default=True, dest="REVB", help = "Reverse the bias corrections")
parser.add_option("-s", "--scan", action = "store_true", default=False, dest="SCAN", help = "Whether to do a scan")
parser.add_option("-q", "--qm", action = "store", type='float', default=-2.46104825e-01, dest="QMS", help = "Qm parameter to scan, add 3 because weird negative number issue")
parser.add_option("-t", "--thickness", action = "store", type='float', default=0.5, dest="THICK", help = "thickness around Qd to scan")
parser.add_option("-w", "--what", action = "store", type="int", default=1, dest="WHICH", help = "Which parameters to scan? 1 for cosmological (qm, qd). 2 for Dipole velocity parameters (qm vs S), 3 for qm qd but cluster job")
parser.add_option("-a", "--acc", action = "store", type="int", default=101, dest="ACC", help = "How many divisions?")
parser.add_option("-i", "--P1L", action = "store", type="float", default=-5, dest="P1L", help = "Parameter 1 lower bound")
parser.add_option("-j", "--P1U", action = "store", type="float", default=-5, dest="P1U", help = "Parameter 1 upper bound")
parser.add_option("-k", "--P2L", action = "store", type="float", default=-5, dest="P2L", help = "Parameter 2 lower bound")
parser.add_option("-l", "--P2U", action = "store", type="float", default=-5, dest="P2U", help = "Parameter 2 upper bound")
(options, args) = parser.parse_args()

options.QMS = options.QMS-3.0
thickness = options.THICK

if options.DET==2:
    STYPE='NoScDep'
elif options.DET==3:
    STYPE='Flat'
elif options.DET==4:
    STYPE='Exp'
elif options.DET==5:
    STYPE='Lin'
else:
    STYPE='None'
    
#elif options.DET==4:
    #velproftype='monolin'


ofname = 'FOuts/OutQKinPhenodL_RH2_D' + str(options.DET)

# Numbers here are hardcoded for the JLA compilation
# The interpolation.npy is only for JLA redshifts

c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

N=740 ; # Number of SNe
CMBdipdec = -7.0
CMBdipra = 168.0


# Spline interpolation of luminosity distance
# Interpolation.npy is a table calculated in Mathematica
# The grid size can be seen from here: .01 between calculated points (in OM-OL space).
# Only calculated for OM in [0,1.5], OL in [-.5,1.5]
interp = np.load( 'Interpolation.npy' )
Z = np.load( 'JLADirZInc.npy' ) ;
Z[:,6][Z[:,6]<0.] = Z[:,6][Z[:,6]<0.] + 360.
jlarr = np.genfromtxt('../jla_likelihood_v6/data/jla_lcparams.txt', skip_header=1)
if options.REVB:
    Z[:,1] = Z[:,1] - jlarr[:,-1]
    ofname = ofname+'RB'
SNG = jlarr.transpose()[17]

def cdAngle(ra1, dec1, ra2, dec2):
    return np.cos(np.deg2rad(dec1))*np.cos(np.deg2rad(dec2))*np.cos(np.deg2rad(ra1) - np.deg2rad(ra2))+np.sin(np.deg2rad(dec1))*np.sin(np.deg2rad(dec2))


#tempInt = [] ;
#for i in range(N):
    #tempInt.append(interpolate.RectBivariateSpline( np.arange(0,1.51,.01), np.arange(-.50,1.51,.01) , interp[i]*(1+  3.77e-4/z[i])))	

def dL( OM, OL ): # Returns in same order as always - c/H0 multiplied on after, in mu
    return np.hstack( [tempdL(OM,OL) for tempdL in tempInt] );
def MU( OM, OL ):
    return 5*np.log10( c/H0 * dL(OM,OL) ) + 25

def MUZ(Zc, Q0, J0):
    k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
    if np.any(np.isnan(k)) or np.any(np.isinf(k)):
        #print 'Fuck', Q0, J0
        k[np.isnan(k)] = 63000.15861331456834
        k[np.isinf(k)] = 630000.15661331456834
    return k

#phenomenological taylor series expansion for dL from Visser et al

def dLPhenoF3(z, q0, j0):
    return z*(1.+0.5*(1.-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0)*z**2.)


def dLPhenoF4(z, q0, j0, s0, a0=1 ,k=0, t=3):
    return z*(1+0.5*(1-q0)*z -1./6.*(1. - q0 - 3.*q0**2. + j0 + k*c**2./(H0**2 * a0**2))*z**2 + 1./24.*(2. -2.*q0 -15.*q0**2 - 15.*q0**3 + 5.*j0 + 10.*q0*j0 + s0 + 2.*k*c**2 * (1.+3.*q0)/(H0**2 * a0**2))*z**3)

#Import JLA data
#cols are z,m,x,c,cluster mass, survey
#Z = np.load( 'JLA.npy' ) ;

#print Z


#Z.transpose()[0] = Z.transpose()[0]*100. + np.random.normal(scale = 3.77e-4, size = len(Z.transpose()[0]))
#### FULL LIKELIHOOD ####

covmatcomponents = [ "cal", "model", "bias", "dust", "sigmalens", "nonia" ]

if not options.PVCO:
    covmatcomponents.append("pecvel")
else:
    ofname = ofname+'_NoPVCov'


ZINDEX=9
if options.FZCMB:
    ofname = ofname + '_FZCMB'
    ZINDEX = 0

COVd = np.load( 'covmat/stat.npy' ) # Constructing data covariance matrix w/ sys.
# sigmaz and sigmalens are constructed as described in the JLA paper
# all others are taken from their .tar and converted to python format
for i in covmatcomponents:
#Notice the lack of "host" covariances - we don't include the mass-step correction.
    COVd += np.load( 'covmat/'+i+'.npy' )

def COV( A , B , VM, VX, VC , RV=0): # Total covariance matrix
    block3 = np.array( [[VM + VX*A**2 + VC*B**2,    -VX*A, VC*B],
                                                [-VX*A , VX, 0],
                                                [ VC*B ,  0, VC]] )
    ATCOVlA = linalg.block_diag( *[ block3 for i in range(N) ] ) ;
    
    if RV==0:
        return np.array( COVd + ATCOVlA );
    elif RV==1:
        return np.array( COVd );
    elif RV==2:
        return np.array( ATCOVlA );

def RES( OM, OL , A , B , M0, X0, C0 ): #Total residual, \hat Z - Y_0*A
    Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ]) 
    mu = MU(OM, OL)[0] ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  


def RESVF3( Q0, J0 , A , B , M0, X0, C0 ): #Total residual, \hat Z - Y_0*A
    Y0A = np.array([ M0-A*X0+B*C0, X0, C0 ]) 
    mu = MUZ(Z[:,0], Q0, J0) ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - Y0A ) for i in range(N) ] )  

def RESVF3RH1( Q0, J0 , A , B , M0, X0, C0, X1, C1, X2, C2, X3, C3 ): #Total residual, \hat Z - Y_0*A
    YA={}
    YA[1.] = np.array([ M0-A*X0+B*C0, X0, C0])
    YA[2.] = np.array([ M0-A*X1+B*C1, X1, C1])
    YA[3.] = np.array([ M0-A*X2+B*C2, X2, C2])
    YA[4.] = np.array([ M0-A*X3+B*C3, X3, C3])
    mu = MUZ(Z[:,0], Q0, J0) ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - YA[SNG[i]] ) for i in range(N)] )  

def RESVF3RH2( Q0, J0 , A , B , M0, X0, C0, X1, C1, X2, C2, X3, C3, SX0, SC0, SX1, SC1, SX2, SC2 ): #Total residual, \hat Z - Y_0*A
    C0 = C0 - SC0 * (Z[:,0] - np.min(Z[:,0][SNG==1]))
    C1 = C1 - SC1 * (Z[:,0] - np.min(Z[:,0][SNG==2]))
    C2 = C2 - SC2 * (Z[:,0] - np.min(Z[:,0][SNG==3]))
    C3 = C3 - 0.*Z[:,0]
    
    X0 = X0 - SX0 * (Z[:,0] - np.min(Z[:,0][SNG==1]))
    X1 = X1 - SX1 * (Z[:,0] - np.min(Z[:,0][SNG==2]))
    X2 = X2 - SX2 * (Z[:,0] - np.min(Z[:,0][SNG==3]))
    X3 = X3 - 0.*Z[:,0]
    
    CF = C0*(SNG==1).astype(float)+C1*(SNG==2).astype(float)+C2*(SNG==3).astype(float)+C3*(SNG==4).astype(float)
    XF = X0*(SNG==1).astype(float)+X1*(SNG==2).astype(float)+X2*(SNG==3).astype(float)+X3*(SNG==4).astype(float)
    
    Y = M0-A*XF + B*CF
    mu = MUZ(Z[:,0], Q0, J0) ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - np.array([Y[i], XF[i], CF[i]])) for i in range(N)] )  


def RESVF3RH2Dip( Q0, J0 , A , B , M0, X0, C0, X1, C1, X2, C2, X3, C3, SX0, SC0, SX1, SC1, SX2, SC2 , QD, DS=np.inf, stype = STYPE): #Total residual, \hat Z - Y_0*A
    C0 = C0 - SC0 * (Z[:,0] - np.min(Z[:,0][SNG==1]))
    C1 = C1 - SC1 * (Z[:,0] - np.min(Z[:,0][SNG==2]))
    C2 = C2 - SC2 * (Z[:,0] - np.min(Z[:,0][SNG==3]))
    C3 = C3 - 0.*Z[:,0]
    
    X0 = X0 - SX0 * (Z[:,0] - np.min(Z[:,0][SNG==1]))
    X1 = X1 - SX1 * (Z[:,0] - np.min(Z[:,0][SNG==2]))
    X2 = X2 - SX2 * (Z[:,0] - np.min(Z[:,0][SNG==3]))
    X3 = X3 - 0.*Z[:,0]
    
    CF = C0*(SNG==1).astype(float)+C1*(SNG==2).astype(float)+C2*(SNG==3).astype(float)+C3*(SNG==4).astype(float)
    XF = X0*(SNG==1).astype(float)+X1*(SNG==2).astype(float)+X2*(SNG==3).astype(float)+X3*(SNG==4).astype(float)
    
    Y = M0-A*XF + B*CF
    cosangle = cdAngle(CMBdipra, CMBdipdec, Z[:,6], Z[:,7])
    Zc = Z[:,ZINDEX]
    if stype=='NoScDep':
        Q = Q0 + QD*cosangle
    elif stype=='Flat':
        #print stype, QD, DS
        Qdip = QD*cosangle
        Qdip[Zc>(DS+0.1)] = 0
        Qdip[Zc>DS] = Qdip[Zc>DS]*np.exp(-1.*(Zc[Zc>DS]-DS)/0.03) #minimizer steps are too small to probe an actual top hat
        Q = Q0 + Qdip
    elif stype=='Exp':
        Qdip = QD*cosangle*np.exp(-1.*Zc/DS)
        #print 'Here', Qdip
        Q = Q0 + Qdip
    elif stype=='Lin':
        Qd = QD - Zc*DS
        Qd[Qd<0] = 0
        Q = Q0 + Qd*cosangle   
    mu = MUZ(Zc, Q, J0) ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - np.array([Y[i], XF[i], CF[i]])) for i in range(N)] )  



def m2loglike(pars , RV = 0):
    if RV != 0 and RV != 1 and RV != 2:
        raise ValueError('Inappropriate RV value')
    else:
        cov = COV( *[ pars[i] for i in [2,5,9,4,7] ] )
        try:
            chol_fac = linalg.cho_factor(cov, overwrite_a = True, lower = True ) 
        except np.linalg.linalg.LinAlgError: # If not positive definite
            return +13993*10.**20 
        except ValueError: # If contains infinity
            return 13995*10.**20
        if options.DET<2:
            res = RESVF3RH2( *[ pars[i] for i in [0,1,2,5,8,3,6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] ] )
        elif options.DET==2:
            res = RESVF3RH2Dip( *[ pars[i] for i in [0,1,2,5,8,3,6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] ] )
        elif options.DET>=3:
            res = RESVF3RH2Dip( *[ pars[i] for i in [0,1,2,5,8,3,6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] ] )
        #Dont throw away the logPI part.
        part_log = 3*N*np.log(2*np.pi) + np.sum( np.log( np.diag( chol_fac[0] ) ) ) * 2
        part_exp = np.dot( res, linalg.cho_solve( chol_fac, res) )

        if pars[0]<-2. or pars[0]>2. or pars[1]<-2. or pars[1]>2. \
            or pars[4]<0 or pars[7]<0 or pars[9]<0:
            part_exp += 100* np.sum(np.array([ _**2 for _ in pars ]))
            # if outside valid region, give penalty

        if RV==0:
            m2loglike = part_log + part_exp
            return m2loglike 
        elif RV==1: 
            return part_exp 
        elif RV==2:
            return part_log 

# Constraint fucntions for fits (constraint is func == 0)

def m2CONSflat( pars ):
    return pars[0] + pars[1] - 1

def m2CONSempt( pars ):
    return pars[0]**2 + pars[1]**2


def m2CONSzm( pars ):
    return pars[0]**2

def m2CONSEdS( pars ):
    return (pars[0]-1)**2 + pars[1]**2

def m2CONSacc( pars ):
    return pars[0]/2. - pars[1]

def m2NoAcc(pars):
    return pars[0]

def m2NODip( pars ):
    return pars[22]

#### CONSTRAINED CHI2 ####

def COV_C( A , B , VM ):
    block1 = np.array( [1 , A , -B] ) ;
    AJLA = linalg.block_diag( *[ block1 for i in range(N) ] );
    return np.dot( AJLA, np.dot( COVd, AJLA.transpose() ) ) + np.eye(N) * VM;

def RES_C( OM, OL, A ,B , M0 ):
    mu = MU(OM,OL)[0] ;
    return Z[:,1] - M0 + A * Z[:,2] - B * Z[:,3] - mu

# INPUT HERE IS REDUCED: pars = [ om, ol, a, b, m0] , VM seperate

def chi2_C( pars, VM ):
    if pars[0]<0 or pars[0]>1.5 or pars[1]<-.50 or pars[1]>1.5 \
        or VM<0:
        return 14994*10.**20
    cov = COV_C( pars[2], pars[3] , VM )
    chol_fac = linalg.cho_factor( cov, overwrite_a = True, lower = True )
    
    res = RES_C( *pars )
    
    part_exp = np.dot( res , linalg.cho_solve( chol_fac, res) )
    return part_exp


bnds = ( (-4.,4.),(-4.,4.),
            (None,None),(None,None),(0,None),
            (None,None),(None,None),(0,None),
            (None,None),(0,None) ,(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))


# Results already found

pre_found_best = np.array([ -3.13971076e-01,   3.72791401e-02,   1.34264121e-01,
         3.80474258e-02,   8.68197288e-01,   3.05785938e+00,
        -1.60647458e-02,   5.04579623e-03,  -1.90471016e+01,
         1.17220199e-02, 3.80474258e-02, -1.60647458e-02,3.80474258e-02, -1.60647458e-02, 3.80474258e-02, -1.60647458e-02, 0., 0.,0., 0.,0., 0.])

pre_found_flat = np.array([   -0.45,   0.5,    1.34605635e-01,
                            3.88748714e-02,   8.67710982e-01,   3.05973830e+00,
                            -1.60202529e-02,   5.04243167e-03,  -1.90547810e+01,
                            1.16957181e-02])

pre_found_empty = np.array([   -0.45,   0.5,     1.32775473e-01,
                            3.35901703e-02,   8.68743215e-01,   3.05069685e+00,
                            -1.47499271e-02,   5.05601201e-03,  -1.90138708e+01,
                            1.19714588e-02])

pre_found_ZM = np.array([  9.86339832e-11,   9.39491731e-02,   1.33690024e-01,
                            3.58225394e-02,   8.69494545e-01,   3.05946289e+00,
                            -1.68347116e-02,   5.07400516e-03,  -1.90319985e+01,
                            1.18656604e-02])

pre_found_EdS = np.array([  9.99999985e-01,   4.19831315e-09,   1.23244919e-01,
                        1.43884399e-02,   8.59506526e-01,   3.03882366e+00,
                        9.26889810e-03,   5.11253864e-03,  -1.88388322e+01,
                        1.55137148e-02])

pre_found_noacc = np.array([  6.84438318e-02,   3.42219159e-02,   1.32357422e-01,
                            3.26703396e-02,   8.67993385e-01,   3.04503841e+00,
                            -1.33181840e-02,   5.04076126e-03,  -1.90062602e+01,
                            1.19991540e-02])

# Check that these are really the minima !

if options.DET ==2:
    bnds = bnds + ((-10., 10.),)
    defBF = [0.1]
    rads = np.linspace(0., 3000., 300)
    pre_found_best = np.hstack([pre_found_best, defBF])

if options.DET ==3:
    bnds = bnds + ((-10., 10.), (0, 1.5),)
    defBF = [0.1, 0.1] 
    rads = np.linspace(0., 10., 300)
    pre_found_best = np.hstack([pre_found_best, defBF])
    
if options.DET ==4:
    bnds = bnds + ((-10., 10.), (0, 1.5),)
    defBF = [-8.0, 0.03]
    rads = np.linspace(0., 10., 300)
    pre_found_best = np.hstack([pre_found_best, defBF])

if options.DET ==5:
    bnds = bnds + ((-10., 10.), (0, 30),)
    defBF = [0.1, 1.3]
    rads = np.linspace(0., 10., 300)
    pre_found_best = np.hstack([pre_found_best, defBF])



estdict=OrderedDict()

# Check that these are really the minima !
estdict['MLE'] = optimize.minimize(m2loglike, pre_found_best, method = 'SLSQP', tol=10**-12,  options={'maxiter':24000}, bounds=bnds)

print 'Basic :', estdict['MLE'] 


    




def DoCosmoScan(resmax, like=m2loglike, prec=options.ACC):
    #SCindices = [0, 11]
    #Othindices = [1,2,3,4,5,6,7,8,9,10, 12]
    print 'Doing likelihood scan for cosmological parameters'
    def cosmolike(pars, QMM=resmax[0], QDD=resmax[-2], RV=0):
        lpars = np.asarray([QMM] + [pars[x] for x in [0, 1,2,3,4,5,6,7,8, 9]] + [QDD] + [pars[-1]])
        return like(lpars, RV=RV)
    llhdict={}
    nllhdict={}
    pardict={}
    allresdict={}
    if (options.P1L > -4)*(options.P1U > -4)*(options.P2L > -4)*(options.P2U > -4):
        QmS = np.linspace(options.P1L, options.P1U, prec)
        QdS = np.linspace(options.P2L, options.P2U, prec)
    else:
        QmS = np.linspace(resmax[0]-0.3,resmax[0]+0.3, prec)
        QdS = np.linspace(resmax[-2]-0.3, resmax[-2]+0.3, prec)
    init = [resmax[x] for x in [1,2,3,4,5,6,7,8, 9, 11]]
    for Qm in QmS:
        llhdict[Qm]={}
        nllhdict[Qm]={}
        pardict[Qm]={}
        allresdict[Qm]={}
        for Qd in QdS:
            def Constrainer1( pars ):
                return pars[0]-Qm
            def Constrainer2( pars ):
                return pars[-2]-Qd
            print 'Now doing Qm, Qd = ', Qm, Qd
            tic = time.clock()
            des = optimize.minimize(m2loglike, estdict['MLE'].x, method = 'SLSQP', constraints = ({'type':'eq', 'fun':Constrainer1}, {'type':'eq', 'fun':Constrainer2}), tol=10**-12, bounds=bnds, options={'maxiter':24000})
            toc = time.clock()
            print des
            print 'Taking ', toc-tic, 'for one minimization'
            pardict[Qm][Qd] = des.x
            llhdict[Qm][Qd] = des.fun
            #nllhdict[Qm][Qd] = like(np.asarray([OM, OL] + [x for x in resmax]))
            nllhdict[Qm][Qd] = like(np.asarray([Qm] + [init[x] for x in [1,2,3,4,5,6,7,8, 9]] + [Qd] + [init[-1]]))
            allresdict[Qm][Qd] = des
            init = des.x
    return pardict, llhdict, nllhdict, allresdict


def DoCosmoScan1Qm(resmax, like=m2loglike, prec=options.ACC):
    llhdict={}
    nllhdict={}
    pardict={}
    allresdict={}
    QdS = np.linspace(resmax[-2]-thickness, resmax[-2]+thickness, prec)
    Qm = options.QMS
    for Qd in QdS:
        def Constrainer1( pars ):
            return pars[0]-Qm
        def Constrainer2( pars ):
            return pars[-2]-Qd
        print 'Now doing Qm, Qd = ', Qm, Qd
        tic = time.clock()
        
        cons = ({'type':'eq', 'fun':Constrainer1}, {'type':'eq', 'fun':Constrainer2})
        
        des = optimize.minimize(m2loglike, estdict['MLE'].x, method = 'SLSQP', constraints = cons, tol=10**-12, bounds=bnds, options={'maxiter':24000})
        toc = time.clock()
        print des
        print 'Taking ', toc-tic, 'for one minimization'
        pardict[Qd] = des.x
        llhdict[Qd] = des.fun
        #nllhdict[Qm][Qd] = like(np.asarray([OM, OL] + [x for x in resmax]))
        nllhdict[Qd] = like(np.asarray([Qm] + [resmax[x] for x in [1,2,3,4,5,6,7,8, 9]] + [Qd] + [resmax[-1]]))
        allresdict[Qd] = des
        init = des.x
    return pardict, llhdict, nllhdict, allresdict


if options.SCAN:
    fhead = 'Scans/QPheno_Det_'+str(options.DET)+'_'+'S'+str(options.WHICH)
    fhead = fhead+'Prec_'+str(options.ACC)+'_'+ofname.replace('/', '_')
    if options.WHICH==1:
        pdict, ldict, ndict, adict = DoCosmoScan(estdict['MLE'].x)
        writedict={}
        writedict['Result'] = estdict['MLE']
        writedict['ScanP'] = pdict
        writedict['ScanL'] = ldict
        writedict['ScanN'] = ndict
        writedict['ScanA'] = adict
        pickle.dump(writedict, open(fhead+'CosmoScan.pickle', "wb"))
    elif options.WHICH==2:
        pdict, ldict, ndict, adict = DoBFScan(res)
        writedict={}
        writedict['Result'] = MLE
        writedict['ScanP'] = pdict
        writedict['ScanL'] = ldict
        writedict['ScanN'] = ndict
        writedict['ScanA'] = adict
        pickle.dump(writedict, open(fhead+'BFModelScan.pickle', "wb"))
    elif options.WHICH==3:
        pdict, ldict, ndict, adict = DoCosmoScan1Qm(estdict['MLE'].x)
        writedict={}
        writedict['Result'] = estdict['MLE']
        writedict['ScanP'] = pdict
        writedict['ScanL'] = ldict
        writedict['ScanN'] = ndict
        writedict['ScanA'] = adict
        writedict['Qm'] = options.QMS
        pickle.dump(writedict, open(fhead+'Qm'+str(options.QMS)+'BFModelScan.pickle', "wb"))





if not options.SCAN:

    estdict['MCENoacc'] = optimize.minimize(m2loglike, pre_found_best, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2NoAcc}, ), tol=10**-12, bounds=bnds, options={'maxiter':24000})

    print "=================="

    print 'No Acc:', estdict['MCENoacc']

    #estdict['MCEflat'] = optimize.minimize(m2loglike, pre_found_flat, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONSflat}, ), tol=10**-12, bounds=bnds)

    #print "=================="

    #print 'Flat:', estdict['MCEflat']

    if options.DET >=2:
        estdict['MCENoDip'] = optimize.minimize(m2loglike, pre_found_best, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2NODip}, ), tol=10**-12, bounds=bnds)
        print "=================="
        print 'No Dipole:', estdict['MCENoDip']

    sqindices = [4, 7, 9]

    fout = open(ofname+'.txt', 'w')

    of = "{0:.4g}"

    for pk in estdict.keys():
        l = pk+'&'+of.format(estdict[pk]['fun'])
        for j in range(len(estdict[pk]['x'])):
            ent = estdict[pk]['x'][j]
            if j in sqindices:
                ent = np.sqrt(ent)
            l = l+'&'+of.format(ent)
        plen = len(estdict[pk]['x'])
        aic = 2.*plen + estdict[pk]['fun']
        aicc = (2.*plen**2. + 2*plen)/(740.-plen-1.)
        l = l+'&'+of.format(aic)+'&'+of.format(aicc)
        l = l+'&'+str(estdict[pk]['success'])
        fout.write(l+'\n')

    fout.close()
