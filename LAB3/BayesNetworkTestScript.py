# -*- coding:utf-8 -*-

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop `Pattern Recognition and Machine Learning` textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])


risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise','long_sit'})
obsVars  = ['smoke', 'exercise','long_sit']
obsVals  = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)


###########################################################################
# Please write your own test script
# HW3 test scripts start from here
###########################################################################

################################ task 1:
# Create factors for the bayesian network
income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'exercise', 'income', 'long_sit'])
bp          = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholest     = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])
stroke       = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
attack      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

############################### task 2
# build the risk net
risk_net = [income, smoke, exercise, long_sit, stay_up, bmi, diabetes, bp, cholest, stroke, attack, angina]
# all the factors
factors = set(riskFactorNet.columns)

print("task2 begins\n")
for disease in ('diabetes', 'stroke', 'attack', 'angina'):
    print(f"task2: {disease} begins")
    margVars = list(factors - {disease, 'smoke', 'exercise','long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals  = [2, 1, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

    margVars = list(factors - {disease, 'bp', 'cholesterol','bmi'})
    obsVars  = ['bp', 'cholesterol','bmi']
    obsVals = [1, 1, 3]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals = [3, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

    print(f"task2: {disease} ends\n\n")
print("task2 completed")


############################### task 3
print("task3 begins")
for i, disease in enumerate(('diabetes', 'stroke', 'attack', 'angina')):
    margVars = list(factors - {disease, 'income'})
    prob = []
    for income in range(1, 9):
        obsVars  = ["income"]
        obsVals  = [income]
        result = inference(risk_net, margVars, obsVars, obsVals)
        prob.append(result["probs"][0])

    plt.subplot(2, 2, i+1)
    plt.plot(np.arange(1, 9), prob)
    plt.xlabel("income-level")
    plt.ylabel(f"Probs of having {disease}")

plt.tight_layout()
plt.savefig("a.png")
print("task3 ends\n\n")

############################### task 4
# Create factors for the bayesian network
income      = readFactorTablefromData(riskFactorNet, ['income'])
new_stroke       = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
new_attack      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
new_angina      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
new_diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])

# build the new risk net
risk_net = [income, smoke, exercise, long_sit, stay_up, bmi, bp, cholest, new_stroke, new_attack, new_angina, new_diabetes]
# all the factors
factors = set(riskFactorNet.columns)


for disease in ('diabetes', 'stroke', 'attack', 'angina'):
    print(f"task4: {disease} begins")
    margVars = list(factors - {disease, 'smoke', 'exercise','long_sit', 'stay_up'})
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals  = [2, 1, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

    margVars = list(factors - {disease, 'bp', 'cholesterol','bmi'})
    obsVars  = ['bp', 'cholesterol','bmi']
    obsVals = [1, 1, 3]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals = [3, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))
    print(f"task4: {disease} completed\n\n")


############################### task 5
print("task5 (without edges): begin")
margVars = list(factors - {"diabetes", "stroke"})
obsVars  = ['diabetes']
obsVals  = [1]
print(inference(risk_net, margVars, obsVars, obsVals))
obsVals  = [3]
print(inference(risk_net, margVars, obsVars, obsVals))
print("task5 (without edges): ends\n\n")


print("task5 (without edges): begin")
new_stroke_2     = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke', 'diabetes'])
risk_net = [income, smoke, exercise, long_sit, stay_up, bmi, new_diabetes, bp, cholest, new_stroke_2, new_attack, new_angina]

margVars = list(factors - {"diabetes", "stroke"})
obsVars  = ['diabetes']
obsVals  = [1]
print(inference(risk_net, margVars, obsVars, obsVals))
obsVals  = [3]
print(inference(risk_net, margVars, obsVars, obsVals))
print("task5 (without edges): ends\n\n")