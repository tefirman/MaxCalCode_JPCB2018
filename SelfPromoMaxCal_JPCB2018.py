# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:47:25 2017

@author: tefirman
"""

""" Importing necessary python modules """

import numpy as np
from scipy.optimize import minimize
import scipy.sparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

""" Specify simulation and inference conditions """

simConditions = 'SelfPromo' ### Descriptive name of your simulation rates ###
timeInc = 300               ### Sampling increment (i.e. number of seconds between frames) ###
numTrials = 100             ### Number of protein expression trajectories to simulate ###
numDays = 7                 ### Length of simulated trajectories in days ###
maxn = 20                   ### Maximum number of RNAs that are allowed in the system ###
maxN = 100                  ### Maximum number of proteins that are allowed in the system ###
maxN2 = 10                  ### Maximum number of dimers that are allowed in the system ###
numIterations = 'Avg'       ### Number of steps to use as "m" in inference; 'Avg' will be reset to average dwell time ###
minMaxl_a = 10              ### Minimum value of "M" to be screened ###
maxMaxl_a = 20              ### Maximum value of "M" to be screened ###
simNum = 1                  ### Simulation Number (in case of testing multiple replicates) ###

def conditionsInitGill(g,g_pro,d,p,r,k_fD,k_bD,k_fP,k_bP,exclusive,\
n_A_init,n_A2_init,n_a_init,n_alpha_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing Gillespie simulation conditions """
    return {'g':g, 'g_pro':g_pro, 'd':d, 'p':p, 'r':r, 'k_fD':k_fD, 'k_bD':k_bD, \
            'k_fP':k_fP, 'k_bP':k_bP, 'exclusive':exclusive, 'N_A_init':n_A_init, \
            'N_A2_init':n_A2_init, 'N_a_init':n_a_init, 'N_alpha_init':n_alpha_init, \
            'inc':inc, 'numSteps':numSteps, 'numTrials':numTrials}

def gillespieSim(conditions,n_A,n_A2,n_a,n_alpha):
    """ Gillespie Simulation of the self-promotion gene circuit in Eqn. (1) """
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_A2 = (conditions['N_A2_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_a = (conditions['N_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        n_alpha = (conditions['N_alpha_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        if len(n_a) < numTrial + 1:
            numA = np.copy(conditions['N_A_init'])
            numA2 = np.copy(conditions['N_A2_init'])
            numa = np.copy(conditions['N_a_init'])
            numalpha = np.copy(conditions['N_alpha_init'])
            n_A.append([np.copy(numA)])
            n_A2.append([np.copy(numA2)])
            n_a.append([np.copy(numa)])
            n_alpha.append([np.copy(numalpha)])
        else:
            numA = np.copy(n_A[numTrial][-1])
            numA2 = np.copy(n_A2[numTrial][-1])
            numa = np.copy(n_a[numTrial][-1])
            numalpha = np.copy(n_alpha[numTrial][-1])
        timeFrame = (len(n_A[numTrial]) - 1)*conditions['inc']
        incCheckpoint = len(n_A[numTrial])*conditions['inc']
        print('Trial #' + str(numTrial + 1))
        while timeFrame < float(conditions['numSteps']*conditions['inc']):
            prob = [conditions['g']*(1 - numalpha),\
            conditions['g_pro']*numalpha,\
            conditions['d']*numa,\
            conditions['p']*numa,\
            conditions['r']*numA,\
            conditions['k_fD']*numA*(numA - 1),\
            conditions['k_bD']*numA2,\
            conditions['k_fP']*(1 - numalpha)*numA2,\
            conditions['k_bP']*numalpha]
            overallRate = sum(prob)
            randNum1 = np.random.rand(1)
            timeFrame -= np.log(randNum1)/overallRate
            while timeFrame >= incCheckpoint:
                n_A[numTrial].append(np.copy(numA).tolist())
                n_A2[numTrial].append(np.copy(numA2).tolist())
                n_a[numTrial].append(np.copy(numa).tolist())
                n_alpha[numTrial].append(np.copy(numalpha).tolist())
                incCheckpoint += conditions['inc']
                if incCheckpoint%10000 == 0:
                    print('Time = ' + str(incCheckpoint) + ' seconds')
            prob = prob/overallRate
            randNum2 = np.random.rand(1)
            if randNum2 <= sum(prob[:2]):
                numa += 1
            elif randNum2 <= sum(prob[:3]):
                numa -= 1
            elif randNum2 <= sum(prob[:4]):
                numA += 1
            elif randNum2 <= sum(prob[:5]):
                numA -= 1
            elif randNum2 <= sum(prob[:6]):
                numA2 += 1
                numA -= 2
            elif randNum2 <= sum(prob[:7]):
                numA2 -= 1
                numA += 2
            elif randNum2 <= sum(prob[:8]):
                numalpha += 1
                numA2 -= 1
            else:
                numalpha -= 1
                numA2 += 1
        n_A[numTrial] = n_A[numTrial][:conditions['numSteps'] + 1]
        n_A2[numTrial] = n_A2[numTrial][:conditions['numSteps'] + 1]
        n_a[numTrial] = n_a[numTrial][:conditions['numSteps'] + 1]
        n_alpha[numTrial] = n_alpha[numTrial][:conditions['numSteps'] + 1]
    return n_A, n_A2, n_a, n_alpha

def peakVals(origHist,numFilter,minVal):
    """ Returns the most probable number of proteins in each state """
    simHist = np.copy(origHist)
    for numTry in range(numFilter):
        for ind in range(1,len(simHist) - 1):
            simHist[ind] = np.sum(simHist[max(ind - 1,0):min(ind + 2,len(simHist))])/\
            np.size(simHist[max(ind - 1,0):min(ind + 2,len(simHist))])
    maxInds = np.where(np.all([simHist[2:] - simHist[1:-1] < 0,\
    simHist[1:-1] - simHist[:-2] > 0,simHist[1:-1] >= minVal],axis=0))[0] + 1
    return maxInds

def entropyStats(n_A,maxInds):
    """ Calculate the path entropies of the trajectory as defined by Eqn. (23) """
    global maxN
    stateProbs = [np.zeros((maxN,maxN)) for ind in range(len(maxInds))]
    cgProbs = np.zeros((len(maxInds),len(maxInds)))
    dwellVals = [[] for ind in range(len(maxInds))]
    for numTrial in range(len(n_A)):
        cgTraj = -1*np.ones(len(n_A[numTrial]))
        for ind in range(len(maxInds)):
            cgTraj[n_A[numTrial] == maxInds[ind]] = ind
        ind1 = np.where(cgTraj >= 0)[0][0]
        inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        while len(inds) > 0:
            stateProbs[int(cgTraj[ind1])] += np.histogram2d(n_A[numTrial][ind1 + 1:ind1 + inds[0]],\
            n_A[numTrial][ind1:ind1 + inds[0] - 1],bins=np.arange(-0.5,maxN))[0]
            cgProbs[int(cgTraj[ind1 + inds[0]]),int(cgTraj[ind1])] += 1
            cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += inds[0]
            dwellVals[int(cgTraj[ind1])].append(inds[0])
            ind1 += inds[0]
            inds = np.where(np.all([cgTraj[ind1:] >= 0,cgTraj[ind1:] != cgTraj[ind1]],axis=0))[0]
        stateProbs[int(cgTraj[ind1])] += np.histogram2d(n_A[numTrial][ind1 + 1:],\
        n_A[numTrial][ind1:-1],bins=np.arange(-0.5,maxN))[0]
        cgProbs[int(cgTraj[ind1]),int(cgTraj[ind1])] += len(n_A[numTrial]) - ind1 - 1
    totProbs = np.zeros((maxN,maxN))
    stateEntropies = []
    for ind in range(len(stateProbs)):
        totProbs += stateProbs[ind]
        stateProbs[ind] = stateProbs[ind]/np.sum(stateProbs[ind])
        stateEntropies.append(-np.nansum(stateProbs[ind]*np.log2(stateProbs[ind])))
    totProbs = totProbs/np.sum(totProbs)
    totEntropy = -np.nansum(totProbs*np.log2(totProbs))
    cgProbs = cgProbs/np.sum(cgProbs)
    macroEntropy = -np.nansum(cgProbs*np.log2(cgProbs))
    return totEntropy, stateEntropies, macroEntropy, dwellVals

def conditionsInitMaxCal(h_a,h_A,k_A,maxl_a,\
n_A_init,l_a_init,l_iA_init,inc,numSteps,numTrials):
    """ Initializes a dictionary containing MaxCal simulation conditions """
    return {'h_a':float(h_a), 'h_A':float(h_A), 'k_A':float(k_A), \
    'maxl_a':maxl_a, 'N_A_init':n_A_init, 'l_a_init':l_a_init, \
    'l_iA_init':l_iA_init, 'inc':inc, 'numSteps':numSteps, \
    'numTrials':numTrials}

def logFactorial(value):
    """ Returns the natural log of the factorial of value using Sterling's Approximation """
    if all([value > 0,abs(round(value) - value) < 0.000001,value <= 34]):
        return float(sum(np.log(range(1,int(value) + 1))))
    elif all([value > 0,abs(round(value) - value) < 0.000001,value > 34]):
        return float(value)*np.log(float(value)) - float(value) + \
        0.5*np.log(2.0*np.pi*float(value)) - 1.0/(12.0*float(value))
    elif value == 0:
        return float(0)
    else:
        return float('nan')

""" Preallocating n-choose-k values so they don't need to be calculated every time """
factMat = []
for ind1 in range(maxN + 1):
    factMat.append([])
    for ind2 in range(ind1 + 1):
        factMat[ind1].append(logFactorial(ind1) - logFactorial(ind2) - logFactorial(ind1 - ind2))
    del ind2
    factMat[ind1] = np.array(factMat[ind1])
del ind1

def transitionProbs(lagrangeVals,maxl_a,n_A):
    """ Returns an array containing transition probabilities as defined by Eqn. (3) """
    global factMat
    logWeight = -float('Inf')*np.ones((n_A + 1,maxl_a + 1))
    templ_a = np.arange(maxl_a + 1)
    for templ_A in range(n_A + 1):
        logWeight[templ_A][templ_a] = factMat[n_A][templ_A] + lagrangeVals[0]*templ_a + \
        lagrangeVals[1]*templ_A + lagrangeVals[2]*templ_a*templ_A
    logWeight = np.exp(logWeight - np.max(np.max(logWeight)))
    logWeight = logWeight/sum(sum(logWeight))
    return logWeight

def maxCalSim(conditions,n_A,l_a,l_iA):
    """ MaxCal Simulation as described in Eqn. (2) and (3) """
    global maxN
    global factMat
    probsTot = [[] for ind in range(maxN)]
    if len(n_A) == 0:
        n_A = (conditions['N_A_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_a = (conditions['l_a_init']*np.ones((conditions['numTrials'],1))).tolist()
        l_iA = (conditions['l_iA_init']*np.ones((conditions['numTrials'],1))).tolist()
    for numTrial in range(max([conditions['numTrials'],len(n_A)])):
        print('Trial #' + str(numTrial + 1))
        if len(n_A) < numTrial + 1:
            n_A.append(np.copy(conditions['N_A_init'][0]).tolist())
            l_a.append(np.copy(conditions['l_a_init'][0]).tolist())
            l_iA.append(np.copy(conditions['l_iA_init'][0]).tolist())
        for numStep in range(len(n_A[numTrial]),len(n_A[numTrial]) + conditions['numSteps']):
            randNum = np.random.rand(1)
            if len(probsTot[n_A[numTrial][numStep - 1]]) == 0:
                probsTot[n_A[numTrial][numStep - 1]] = transitionProbs([conditions['h_a'],\
                conditions['h_A'],conditions['k_A']],conditions['maxl_a'],n_A[numTrial][numStep - 1])
            probSum = 0
            l_aVal = -1
            while l_aVal < conditions['maxl_a'] and randNum > probSum:
                l_aVal += 1
                l_iAVal = n_A[numTrial][numStep - 1] + 1
                while l_iAVal > 0 and randNum > probSum:
                    l_iAVal -= 1
                    probSum += probsTot[n_A[numTrial][numStep - 1]][l_iAVal,l_aVal]
            n_A[numTrial].append(l_iAVal + l_aVal)
            l_a[numTrial].append(l_aVal)
            l_iA[numTrial].append(l_iAVal)
    return n_A, l_a, l_iA

def maxCalFSP(lagrangeVals,maxl_a,n_A_init,numSteps):
    """ Discretized Finite State Projection of Maximum Caliber """
    """ Analytical way of calculating protein number distributions """
    global maxN
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    probInit = np.zeros(len(probMatrix))
    probInit[n_A_init] = 1.0
    return np.dot(np.linalg.matrix_power(probMatrix,numSteps),probInit)

def maxCalEquil(lagrangeVals,maxl_a):
    """ Returns the equilibrium protein number distribution of MaxCal """
    global maxN
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix -= np.identity(probMatrix.shape[0])
    vals,equilProb = np.linalg.eig(probMatrix)
    equilProb = equilProb[:,np.imag(vals) == 0]
    vals = abs(np.real(vals[np.imag(vals) == 0]))
    equilProb = abs(np.real(equilProb[:,np.where(vals == min(vals))[0][0]]))
    return equilProb/sum(equilProb)

def dwellHistMaxCal(conditions,maxInds,maxTol,inc):
    """ Returns analytical dwell time probability distribution of MaxCal """
    global maxN
    dwellProbs = []
    for startInd in range(len(maxInds)):
        probMatrix = np.zeros((maxN,maxN))
        for probN in range(maxN):
            if probN in maxInds and probN != maxInds[startInd]:
                continue
            probsMC = transitionProbs([conditions['h_a'],conditions['h_A'],\
            conditions['k_A']],conditions['maxl_a'],probN).reshape((probN + 1)*(conditions['maxl_a'] + 1))
            finalN = np.array([np.arange(ind,ind + conditions['maxl_a'] + 1) \
            for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(conditions['maxl_a'] + 1))
            probsMC = probsMC[finalN < maxN]
            finalN = finalN[finalN < maxN]
            for ind in range(len(finalN)):
                probMatrix[finalN[ind],probN] += probsMC[ind]
        probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
        probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
        probMatrix = np.linalg.matrix_power(probMatrix,inc)
        probInit = np.zeros(len(probMatrix))
        probInit[maxInds[startInd]] = 1.0
        dwellCumeProb = [0.0]
        while 1 - dwellCumeProb[-1] > maxTol:
            probInit = np.dot(probMatrix,probInit)
            dwellCumeProb.append(sum(probInit[maxInds]) + probInit[-1] - probInit[maxInds[startInd]])
        dwellProbs.append(np.array(dwellCumeProb[1:]) - np.array(dwellCumeProb[:-1]))
    return dwellProbs

def rateCalc(lagrangeVals,maxl_a,n_A):
    """ Calculates effective production and degradation rates of MaxCal parameters as defined in Eqn. (4) """
    l_A = np.array([indA*np.ones(maxl_a + 1) for indA in range(n_A + 1)])
    l_a = np.array([np.arange(maxl_a + 1) for indA in range(n_A + 1)])
    probVals = transitionProbs(lagrangeVals,maxl_a,n_A)
    prodRateA = np.sum(l_a*probVals)
    if n_A > 0:
        degRateA = np.sum((n_A - l_A)*probVals)
        degRateA /= n_A
    else:
        degRateA = float('NaN')
    return prodRateA, degRateA

def maxCal_mle(lagrangeVals):
    """ Calculates the negative natural log of the trajectory likelihood given specified MaxCal parameters """
    global probs
    global maxl_a
    global maxN
    global numIterations
    probMatrix = np.zeros((maxN,maxN))
    for probN in range(maxN):
        probsMC = transitionProbs(lagrangeVals,maxl_a,probN).reshape((probN + 1)*(maxl_a + 1))
        finalN = np.array([np.arange(ind,ind + maxl_a + 1) \
        for ind in range(probN + 1)],dtype=int).reshape((probN + 1)*(maxl_a + 1))
        probsMC = probsMC[finalN < maxN]
        finalN = finalN[finalN < maxN]
        for ind in range(len(finalN)):
            probMatrix[finalN[ind],probN] += probsMC[ind]
    probMatrix = np.hstack((probMatrix,np.zeros((maxN,1))))
    probMatrix = np.vstack((probMatrix,1 - probMatrix.sum(axis=0)))
    tempMatrix = np.linalg.matrix_power(probMatrix,numIterations)
    loglike = -1*np.nansum(np.nansum(np.log(tempMatrix)*probs.toarray()))
#    print('h_a = ' + str(round(lagrangeVals[0],3)) + ', h_A = ' + \
#    str(round(lagrangeVals[1],3)) + ', K_A = ' + str(round(lagrangeVals[2],4)) + \
#    ', loglike = ' + str(round(loglike,1)))
    return loglike

""" Defining Gillespie Parameters """
#conditions_Gill = conditionsInitGill(g,g_pro,d,p,r,k_fD,k_bD,k_fP,k_bP,exclusive,\
#n_A_init,n_A2_init,n_a_init,n_alpha_init,inc,numSteps,numTrials)
conditions_Gill = conditionsInitGill(0.05,0.5,0.2,0.02,0.001,0.005,50.0,\
0.006,0.00003,False,5,0,5,0,timeInc,int((24*3600*numDays)/timeInc),numTrials)

""" Loading/Simulating Expression Trajectories """

if os.path.exists(simConditions + str(simNum) + '.npz'):
    """ If the simulation already exists, load it and sample at specified rate and length """
    tempVars = np.load(simConditions + str(simNum) + '.npz')
    n_A_Gill = tempVars['n_A_Gill']
    n_A_Gill = n_A_Gill[:,range(0,24*3600*numDays + 1,timeInc)]
    del tempVars
else:
    """ Running a few trials to get a basic idea of the equilibrium distribution """
    conditions_Gill['numTrials'] = 2
    n_A_temp,n_A2_temp,n_a_temp,n_alpha_temp = gillespieSim(conditions_Gill,[],[],[],[])
    equilProb = np.zeros((2,maxn,maxN2,maxN))
    for numTrial in range(len(n_A_temp)):
        print('Trial #' + str(numTrial + 1))
        for numStep in range(len(n_A_temp[numTrial])):
            equilProb[int(n_alpha_temp[numTrial][numStep]),\
                      int(n_a_temp[numTrial][numStep]),\
                      int(n_A2_temp[numTrial][numStep]),\
                      int(n_A_temp[numTrial][numStep])] += 1
        del numStep
    del numTrial
    equilProb = equilProb/np.sum(equilProb)
    del n_A_temp
    del n_A2_temp
    del n_a_temp
    del n_alpha_temp
    conditions_Gill['numTrials'] = numTrials
    
    """ Randomly picking starting conditions that are representative of equilibrium """
    n_A_Gill = []
    n_A2_Gill = []
    n_a_Gill = []
    n_alpha_Gill = []
    for numTrial in range(numTrials):
        print('Trial #' + str(numTrial + 1))
        randNum = np.random.uniform(low=0,high=1)
        totProb = 0.0
        for n_alpha_init in range(2):
            for n_a_init in range(maxn):
                for n_A2_init in range(maxN2):
                    for n_A_init in range(maxN):
                        totProb += equilProb[n_alpha_init,n_a_init,n_A2_init,n_A_init]
                        if totProb > randNum:
                            break
                    if totProb > randNum:
                        break
                if totProb > randNum:
                    break
            if totProb > randNum:
                break
        n_A_Gill.append([n_A_init])
        n_A2_Gill.append([n_A2_init])
        n_a_Gill.append([n_a_init])
        n_alpha_Gill.append([n_alpha_init])
        del n_A_init
        del n_A2_init
        del n_a_init
        del n_alpha_init
        del totProb
        del randNum
    del numTrial
    del equilProb
    
    """ Running simulations at one second per frame and saving as npz file """
    conditions_Gill['inc'] = 1
    conditions_Gill['numSteps'] = int(24*3600*numDays/conditions_Gill['inc'])
    n_A_Gill,n_A2_Gill,n_a_Gill,n_alpha_Gill = gillespieSim(conditions_Gill,n_A_Gill,n_A2_Gill,n_a_Gill,n_alpha_Gill)
    np.savez_compressed(simConditions + str(simNum) + '.npz', conditions_Gill=conditions_Gill, \
    n_A_Gill=n_A_Gill, n_A2_Gill=n_A2_Gill, n_a_Gill=n_a_Gill, n_alpha_Gill=n_alpha_Gill)
    
    """ Sampling the trajectory at the specified rate and length """
    n_A_Gill = np.array(n_A_Gill)[:,range(0,24*3600*numDays + 1,timeInc)]
    conditions_Gill['inc'] = timeInc
    conditions_Gill['numSteps'] = int((24*3600*numDays)/timeInc)
    del n_A2_Gill
    del n_a_Gill
    del n_alpha_Gill

""" Assessing Gillespie Simulations """

""" Protein Number Distribution """
simHist_Gill = np.histogram(n_A_Gill,bins=np.arange(-0.5,maxN))[0]
simHist_Gill = simHist_Gill/sum(simHist_Gill)
maxInds_Gill = peakVals(simHist_Gill[:75],5,0.01)
""" Path Entropy Statistics """
totEntropy_Gill,stateEntropies_Gill,macroEntropy_Gill,dwellVals_Gill = entropyStats(n_A_Gill,maxInds_Gill)
""" Dwell Time Statistics """
avgDwells_Gill = []
avgTotDwell_Gill = []
for ind in range(len(dwellVals_Gill)):
    avgDwells_Gill.append(np.average(dwellVals_Gill[ind])*conditions_Gill['inc'])
    avgTotDwell_Gill.extend(dwellVals_Gill[ind])
del ind
avgTotDwell_Gill = np.average(avgTotDwell_Gill)*conditions_Gill['inc']
""" Resetting "m" to average dwell time if specified at the top """
if type(numIterations) == str:
    numIterations = int(round(avgTotDwell_Gill/conditions_Gill['inc']))
""" Counting observed transitions for likelihood calculations (omega_i->j in Eqn. 7) """
probs = scipy.sparse.csc_matrix((np.ones(np.size(n_A_Gill[:,numIterations::numIterations])),\
(n_A_Gill[:,numIterations::numIterations].reshape(np.size(n_A_Gill[:,numIterations::numIterations])),\
n_A_Gill[:,:-numIterations:numIterations].reshape(np.size(n_A_Gill[:,:-numIterations:numIterations])))),shape=(maxN + 1,maxN + 1))
del n_A_Gill

""" Maximizing Likelihood with respect to Lagrange Multipliers for each value of M """

loglike = float('NaN')*np.ones(maxMaxl_a)
finalGuess = float('NaN')*np.ones((maxMaxl_a,3))
for maxl_a in range(minMaxl_a,maxMaxl_a + 1):
    print('Max l_a = ' + str(maxl_a))
    if maxl_a == minMaxl_a:
        """ Start with a random guess... """
        bestGuess = [np.random.uniform(low=-2,high=-1),np.random.uniform(low=1,high=2),0.03]
    else:
        """ ... then use the previous M values final solution """
        bestGuess = finalGuess[maxl_a - 2]
    """ Technically minimizing the negative log likelihood """
    res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
    tol=0.1,options={'disp':True,'maxiter':500})
    """ Storing results """
    loglike[maxl_a - 1] = res['fun']
    finalGuess[maxl_a - 1] = res['x']
""" Saving results thus far """
np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)

""" Sweeping the phase space of M back and forth for higher confidence in results """
for numTry in range(2):
    for maxl_a in range(maxMaxl_a - 1,minMaxl_a - 1,-1):
        print('Max l_a = ' + str(maxl_a))
        """ Use the final solution of M+1 with a random kick """
        bestGuess = finalGuess[maxl_a] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01])
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
        tol=0.1,options={'disp':True,'maxiter':500})
        if res['fun'] < loglike[maxl_a - 1]:
            """ Storing results if they are better """
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
    """ Saving results thus far """
    np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
    str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
    '_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
    conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)
    
    for maxl_a in range(minMaxl_a + 1,maxMaxl_a + 1):
        print('Max l_a = ' + str(maxl_a))
        """ Use the final solution of M-1 with a random kick """
        bestGuess = finalGuess[maxl_a - 2] + np.array([0.2*np.random.rand(1)[0] - 0.1,\
        0.2*np.random.rand(1)[0] - 0.1,0.02*np.random.rand(1)[0] - 0.01])
        res = minimize(maxCal_mle,bestGuess,method='nelder-mead',\
        tol=0.1,options={'disp':True,'maxiter':500})
        if res['fun'] < loglike[maxl_a - 1]:
            """ Storing results if they are better """
            loglike[maxl_a - 1] = res['fun']
            finalGuess[maxl_a - 1] = res['x']
    """ Saving results thus far """
    np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
    str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
    '_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
    conditions_Gill=conditions_Gill, probs=probs, loglike=loglike, finalGuess=finalGuess)
del numTry

""" Plotting the negative log likelihood as a function of M for a sanity check """
fig = plt.figure()
plt.plot(range(1,len(loglike) + 1),loglike)
plt.xlabel('Max l_alpha')
plt.ylabel('Negative Log Likelihood')
plt.grid(True)
plt.savefig('LogLikeVsMaxl_a_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.pdf')
plt.close(fig)
del fig

""" Selecting the most likely value of M and assessing the fit """

maxl_a = int(np.where(loglike == np.nanmin(loglike))[0][0]) + 1
bestGuess = finalGuess[maxl_a - 1]
""" Protein Number Distribution """
equilProb_MaxCal = maxCalEquil(bestGuess,maxl_a)
maxInds_MaxCal = peakVals(equilProb_MaxCal[:75],5,0.01)
""" Effective Production and Degradation Rates """
prodRates = float('NaN')*np.ones(len(maxInds_MaxCal))
degRates = float('NaN')*np.ones(len(maxInds_MaxCal))
for ind in range(len(maxInds_MaxCal)):
    prodRates[ind],degRates[ind] = rateCalc(bestGuess,maxl_a,maxInds_MaxCal[ind])
del ind
prodRates /= timeInc
degRates /= timeInc

""" Plotting protein number distributions of Gillespie and MaxCal to compare """

mpl.rc('font',family='Arial')
mpl.rc('font',size=12)
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
plt.rc('font',weight='bold')

fig = plt.figure()
ax = fig.gca()
plt.plot(range(len(simHist_Gill)),simHist_Gill,'b',linewidth=2.0)
plt.plot(range(len(equilProb_MaxCal)),equilProb_MaxCal,'r',linewidth=2.0)
#plt.grid(True)
ax.text(6,0.075,'A',fontsize=30,fontweight='bold')
plt.axis([0,80,0,0.09])
plt.xticks(np.arange(0,81,20))
plt.yticks(np.arange(0.0,0.081,0.02))
plt.xlabel('# of proteins',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('ProteinNumberDistribution_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.pdf')

""" Saving fitting assessments thus far """

np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates)

""" Storing most likely Maximum Caliber parameters for simulation """
#conditions_MaxCal = conditionsInitMaxCal(h_a,h_A,k_A,maxl_a,\
#n_A_init,l_a_init,l_iA_init,inc,numSteps,numTrials)
conditions_MaxCal = conditionsInitMaxCal(bestGuess[0],bestGuess[1],bestGuess[2],maxl_a,\
5,0,5,timeInc,int((24*3600*numDays)/timeInc),numTrials)
""" Randomly picking starting conditions that are representative of equilibrium """
n_A_MaxCal = []
for numTrial in range(numTrials):
    randNum = np.random.uniform(low=0,high=1)
    totProb = 0.0
    for n_A_init in range(maxN):
        totProb += equilProb_MaxCal[n_A_init]
        if totProb > randNum:
            break
    n_A_MaxCal.append([n_A_init])
    del n_A_init
    del totProb
    del randNum
del numTrial
l_a_MaxCal = np.zeros((conditions_MaxCal['numTrials'],1)).tolist()
l_iA_MaxCal = np.copy(n_A_MaxCal).tolist()
""" Running a simulation with the given sampling rate and length """
n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal = maxCalSim(conditions_MaxCal,n_A_MaxCal,l_a_MaxCal,l_iA_MaxCal)
""" Calculating MaxCal path entropy and dwell time statistics for comparison with Gillespie """
totEntropy_MaxCal,stateEntropies_MaxCal,macroEntropy_MaxCal,dwellVals_MaxCal = entropyStats(n_A_MaxCal,maxInds_MaxCal)
avgDwells_MaxCal = []
avgTotDwell_MaxCal = []
for ind in range(len(dwellVals_MaxCal)):
    avgDwells_MaxCal.append(np.average(dwellVals_MaxCal[ind])*conditions_MaxCal['inc'])
    avgTotDwell_MaxCal.extend(dwellVals_MaxCal[ind])
del ind
avgTotDwell_MaxCal = np.average(avgTotDwell_MaxCal)*conditions_MaxCal['inc']

""" Saving fitting assessments thus far """

np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates, conditions_MaxCal=conditions_MaxCal, \
n_A_MaxCal=n_A_MaxCal, l_a_MaxCal=l_a_MaxCal, l_iA_MaxCal=l_iA_MaxCal, \
stateEntropies_MaxCal=stateEntropies_MaxCal, macroEntropy_MaxCal=macroEntropy_MaxCal, \
totEntropy_MaxCal=totEntropy_MaxCal, dwellVals_MaxCal=dwellVals_MaxCal, \
avgDwells_MaxCal=avgDwells_MaxCal, avgTotDwell_MaxCal=avgTotDwell_MaxCal)

""" Calculating dwell time distributions from simulations """
dwellInds_Gill = [[] for ind in range(len(dwellVals_Gill))]
dwellProbs_Gill = [[] for ind in range(len(dwellVals_Gill))]
for ind in range(len(dwellVals_Gill)):
    dwellProbs_Gill[ind],dwellInds_Gill[ind] = np.histogram(np.array(dwellVals_Gill[ind])*conditions_Gill['inc'],np.arange(0,300000,9000))
    dwellProbs_Gill[ind] = dwellProbs_Gill[ind]/sum(dwellProbs_Gill[ind])
del ind
dwellInds_MaxCalSim = [[] for ind in range(len(dwellVals_MaxCal))]
dwellProbs_MaxCalSim = [[] for ind in range(len(dwellVals_MaxCal))]
for ind in range(len(dwellVals_MaxCal)):
    dwellProbs_MaxCalSim[ind],dwellInds_MaxCalSim[ind] = np.histogram(np.array(dwellVals_MaxCal[ind])*conditions_MaxCal['inc'],np.arange(0,300000,9000))
    dwellProbs_MaxCalSim[ind] = dwellProbs_MaxCalSim[ind]/sum(dwellProbs_MaxCalSim[ind])
del ind

""" Plotting low-to-high dwell time distributions to compare Gillespie and MaxCal """
fig = plt.figure()
ax = fig.gca()
plt.plot(dwellInds_Gill[0][:-1],dwellProbs_Gill[0],'b',linewidth=2.0)
plt.plot(dwellInds_MaxCalSim[0][:-1],dwellProbs_MaxCalSim[0],'r',linewidth=2.0)
#plt.grid(True)
ax.text(15000,0.115,'B',fontsize=30,fontweight='bold')
plt.axis([0,300000,0,0.14])
plt.xticks(np.arange(0,300001,100000))
plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
ax.xaxis.major.formatter._useMathText = True
plt.yticks(np.arange(0,0.14,0.04))
plt.xlabel('Low-to-High Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('LowToHighDwellDistribution_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.pdf')

""" Plotting high-to-low dwell time distributions to compare Gillespie and MaxCal """
fig = plt.figure()
ax = fig.gca()
plt.plot(dwellInds_Gill[1][:-1],dwellProbs_Gill[1],'b',linewidth=2.0)
plt.plot(dwellInds_MaxCalSim[1][:-1],dwellProbs_MaxCalSim[1],'r',linewidth=2.0)
#plt.grid(True)
ax.text(15000,0.115,'C',fontsize=30,fontweight='bold')
plt.axis([0,300000,0,0.14])
plt.xticks(np.arange(0,300001,100000))
plt.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
ax.xaxis.major.formatter._useMathText = True
plt.yticks(np.arange(0,0.14,0.04))
plt.xlabel('High-to-Low Dwell Time (s)',fontsize=18,fontweight='bold')
plt.ylabel('Probability',fontsize=18,fontweight='bold')
#plt.legend(['Gillespie','MaxCal'],fontsize=15)
plt.tight_layout()
plt.savefig('HighToLowDwellDistribution_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.pdf')

""" Saving fitting assessments """

np.savez_compressed('ExtractedParameters_SelfPromoMaxCal_' + simConditions + \
str(simNum) + '_Trials' + str(numTrials) + '_Days' + str(numDays) + \
'_Inc' + str(timeInc) + '_Iter' + str(numIterations) + '.npz', \
conditions_Gill=conditions_Gill, simHist_Gill=simHist_Gill, \
maxInds_Gill=maxInds_Gill, stateEntropies_Gill=stateEntropies_Gill, \
macroEntropy_Gill=macroEntropy_Gill, totEntropy_Gill=totEntropy_Gill, \
dwellVals_Gill=dwellVals_Gill, avgDwells_Gill=avgDwells_Gill, \
avgTotDwell_Gill=avgTotDwell_Gill, probs=probs, loglike=loglike, \
finalGuess=finalGuess, maxl_a=maxl_a, bestGuess=bestGuess, \
equilProb_MaxCal=equilProb_MaxCal, maxInds_MaxCal=maxInds_MaxCal, \
prodRates=prodRates, degRates=degRates, conditions_MaxCal=conditions_MaxCal, \
n_A_MaxCal=n_A_MaxCal, l_a_MaxCal=l_a_MaxCal, l_iA_MaxCal=l_iA_MaxCal, \
stateEntropies_MaxCal=stateEntropies_MaxCal, macroEntropy_MaxCal=macroEntropy_MaxCal, \
totEntropy_MaxCal=totEntropy_MaxCal, dwellVals_MaxCal=dwellVals_MaxCal, \
avgDwells_MaxCal=avgDwells_MaxCal, avgTotDwell_MaxCal=avgTotDwell_MaxCal, \
dwellInds_MaxCalSim=dwellInds_MaxCalSim, dwellProbs_MaxCalSim=dwellProbs_MaxCalSim)







