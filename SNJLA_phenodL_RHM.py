import numpy as np
from scipy import interpolate, linalg, optimize
from collections import OrderedDict

# Numbers here are hardcoded for the JLA compilation
# The interpolation.npy is only for JLA redshifts

c = 299792.458 # km/s
H0 = 70 #(km/s) / Mpc

N=740 ; # Number of SNe

# Spline interpolation of luminosity distance
# Interpolation.npy is a table calculated in Mathematica
# The grid size can be seen from here: .01 between calculated points (in OM-OL space).
# Only calculated for OM in [0,1.5], OL in [-.5,1.5]
interp = np.load( 'Interpolation.npy' )
Z = np.load( 'JLA.npy' ) ;
z=Z.transpose()[0]
jlarr = np.genfromtxt('../jla_likelihood_v6/data/jla_lcparams.txt', skip_header=1)
SNG = jlarr.transpose()[17]


tempInt = [] ;
for i in range(N):
    tempInt.append(interpolate.RectBivariateSpline( np.arange(0,1.51,.01), np.arange(-.50,1.51,.01) , interp[i]*(1+  3.77e-4/z[i])))	

def dL( OM, OL ): # Returns in same order as always - c/H0 multiplied on after, in mu
    return np.hstack( [tempdL(OM,OL) for tempdL in tempInt] );
def MU( OM, OL ):
    return 5*np.log10( c/H0 * dL(OM,OL) ) + 25

def MUZ(Zc, Q0, J0):
    k = 5.*np.log10( c/H0 * dLPhenoF3(Zc, Q0, J0)) + 25.   
    if np.any(np.isnan(k)):
        print 'Fuck', Q0, J0
        k[np.isnan(k)] = 63.15861331456834
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

#Z=Z*0.
#print Z.transpose()[0]


#print Z

COVd = np.load( 'covmat/stat.npy' ) # Constructing data covariance matrix w/ sys.
# sigmaz and sigmalens are constructed as described in the JLA paper
# all others are taken from their .tar and converted to python format
for i in [ "cal", "model", "bias", "dust", "pecvel", "sigmaz", "sigmalens", "nonia" ]:
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

def RESVF3RH1M( Q0, J0 , A , B , M0, X0, C0, X1, C1, X2, C2, X3, C3, M1, M2, M3 ): #Total residual, \hat Z - Y_0*A                                        
    YA={}
    YA[1.] = np.array([ M0-A*X0+B*C0, X0, C0])
    YA[2.] = np.array([ M1-A*X1+B*C1, X1, C1])
    YA[3.] = np.array([ M2-A*X2+B*C2, X2, C2])
    YA[4.] = np.array([ M3-A*X3+B*C3, X3, C3])
    mu = MUZ(Z[:,0], Q0, J0) ;
    return np.hstack( [ (Z[i,1:4] -np.array([mu[i],0,0]) - YA[SNG[i]] ) for i in range(N)] )

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
        res = RESVF3RH1M( *[ pars[i] for i in [0,1,2,5,8,3,6, 10, 11, 12, 13, 14, 15, 16, 17, 18] ] )

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
            (None,None),(0,None) ,(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))


# Results already found

pre_found_best = np.array([ -3.13971076e-01,   3.72791401e-02,   1.34264121e-01,
         3.80474258e-02,   8.68197288e-01,   3.05785938e+00,
        -1.60647458e-02,   5.04579623e-03,  -1.90471016e+01,
         1.17220199e-02, 3.80474258e-02, -1.60647458e-02,3.80474258e-02, -1.60647458e-02, 3.80474258e-02, -1.60647458e-02, -1.90471016e+01, -1.90471016e+01, -1.90471016e+01])

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

estdict=OrderedDict()

estdict['MLE'] = optimize.minimize(m2loglike, pre_found_best, method = 'SLSQP', tol=10**-12,  options={'maxiter':24000}, bounds=bnds)

estdict['MCENoacc'] = optimize.minimize(m2loglike, pre_found_best, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2NoAcc}, ), tol=10**-12, bounds=bnds, options={'maxiter':24000})


print "=================="

print 'No Acc:', estdict['MCENoacc']

sqindices = [4, 7, 9]

ofname = 'FOuts/QPhenoRHM'

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


#MCEflat = optimize.minimize(m2loglike, pre_found_flat, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONSflat}, ), tol=10**-10)
#MCEempty = optimize.minimize(m2loglike, pre_found_empty, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONSempt}, ), tol=10**-10)
#MCEzeromatter = optimize.minimize(m2loglike, pre_found_ZM, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONSzm}, ), tol=10**-10)
#MCEEdS = optimize.minimize(m2loglike, pre_found_EdS, method = 'SLSQP', constraints = ({'type':'eq', 'fun':m2CONSEdS}, ), tol=10**-10)

#, MCEflat , MCEempty , MCEzeromatter , MCEEdS

