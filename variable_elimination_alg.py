# Hello World program in Python
    
# Yan Feng
# Assignment 3 Q1 variable elimination alg

import numpy as np

class Factor:

    def __init__(self, varlist, vallist, flist):
        self.varlist = list(varlist)
        self.vallist = list(vallist) #this list can be omitted
        self.flist = flist.copy()
        self.helperTable = dict();
        for i in range(len(varlist)):
            self.helperTable[varlist[i]] = vallist[i]

    def selectionSort(self):
        for i in range(0, len(self.varlist)):
            min = i;
            for j in range(i+1, len(self.varlist)):
                if(self.varlist[j] < self.varlist[min]):
                    min = j
            #swap
            temp = self.varlist[i]
            self.varlist[i] = self.varlist[min]
            self.varlist[min] = temp
            self.flist = np.swapaxes(self.flist, i, min)

    @staticmethod
    def restrict(factor, variable, value):
        varindex = factor.varlist.index(variable)
        valindex = factor.helperTable[variable].index(value)
        newfactor = Factor(factor.varlist, factor.vallist, factor.flist)
        del newfactor.varlist[varindex]
        del newfactor.vallist[varindex]
        del newfactor.helperTable[variable]
        format = ()
        for i in range(varindex):
            format += slice(None),
        format += slice(valindex, valindex + 1),
        newfactor.flist = newfactor.flist[format]
        reshapeformat = ()
        for val in newfactor.vallist:
            reshapeformat += (len(val),)
        newfactor.flist = newfactor.flist.reshape(reshapeformat)
        return newfactor

    @staticmethod
    def multiply(factor1, factor2):
        #redesign factor1 and factor2 so that the variables follow the same order
        factor1.selectionSort()
        factor2.selectionSort()
        #find the union of both factor variables
        unionV = list(factor1.varlist)
        for i in range(0, len(factor2.varlist)):
            if(factor2.varlist[i] not in unionV):
                unionV.append(factor2.varlist[i])
        unionV.sort()
        #expand both factors that have same number of dimensions
        f1_varlist = list(factor1.varlist)
        f2_varlist = list(factor2.varlist)
        f1_flist = factor1.flist.copy()
        f2_flist = factor2.flist.copy()
        for j in range(0, len(unionV)):
            if(unionV[j] not in f1_varlist) or (f1_varlist[j] != unionV[j]):
                f1_varlist.insert(j, unionV[j])
                f1_flist = np.expand_dims(f1_flist, axis=j)
            if(unionV[j] not in f2_varlist) or (f2_varlist[j] != unionV[j]):
                f2_varlist.insert(j, unionV[j])
                f2_flist = np.expand_dims(f2_flist, axis=j)
        #construct new Factor as product, multiply factor1 and factor2
        product_varlist = list(unionV)
        product_vallist = list()
        product_flist = f1_flist*f2_flist
        for k in range(0, len(unionV)):
            product_vallist.append(['t', 'f'])
        product = Factor(product_varlist, product_vallist, product_flist)
        return product

    @staticmethod
    def sumOut(factor, variable):
        varindex = factor.varlist.index(variable)
        newfactor = Factor(factor.varlist, factor.vallist, factor.flist)
        del newfactor.varlist[varindex]
        del newfactor.vallist[varindex]
        del newfactor.helperTable[variable]
        newfactor.flist = np.sum(newfactor.flist, axis=varindex)
        return newfactor

    @staticmethod
    def normalize(factor):
        newfactor = Factor(factor.varlist, factor.vallist, factor.flist)
        #divide by sum of of probabilities
        sumf = np.sum(newfactor.flist)
        newfactor.flist = np.true_divide(newfactor.flist, sumf)
        return newfactor

    @staticmethod
    def inference(factorlist, querylist, evidencelist, hiddenvarlist):
        #set the observed variable to their observed values
        print('Step 2: Restrict the factors')
        for i in range(0, len(factorlist)):
            for observedvar, observedval in evidencelist:
                if (observedvar in factorlist[i].varlist):
                    print(' =>For evidence ' + observedvar +  ', we restrict factor(' + ','.join(factorlist[i].varlist) + ') to ', end='')
                    factorlist[i] = Factor.restrict(factorlist[i], observedvar, observedval)
                    print('factor(' + ','.join(factorlist[i].varlist) + ') = ', end='')
                    print(factorlist[i].flist)
        #printFactors(factorlist)

        #for each of the hidden variable, sum out the variable
        print('Step 3: Sum out each hidden variable')
        for hiddenvar in hiddenvarlist:
            factorContainingHiddenV = list()
            factorNotConatinHiddenV = list()
            for factor in factorlist:
                if(hiddenvar in factor.varlist):
                    factorContainingHiddenV.append(factor)
                else:
                    factorNotConatinHiddenV.append(factor)
            print(' =>We will sum out ' + hiddenvar)
            #multiply the factors together
            multi = False
            product = factorContainingHiddenV[0]
            for fh in factorContainingHiddenV[1:]:
                multi = True
                product = Factor.multiply(product, fh)
            if multi:
                print('     Multiply the factors together, we get factor(' + ','.join(product.varlist) + ') = ', end='')
                print(product.flist)
            #sum out the current hidden variavble from the product
            sumOutResult = Factor.sumOut(product, hiddenvar)
            print('     Sum out ' + hiddenvar + ', we define a new factor(' + ','.join(sumOutResult.varlist) + ') = ', end='' )
            print(sumOutResult.flist)
            factorNotConatinHiddenV.append(sumOutResult)
            factorlist = factorNotConatinHiddenV
        #printFactors(factorlist)

        #Multiple the remaining factors
        print('Step 4: Multiple the remaining factors')
        product = factorlist[0]
        for ftr in factorlist[1:]:
            product = Factor.multiply(product, ftr)
        printFactors([product])
        
        #Normalize by dividing the resulting factor
        print('Step 5: Normalize the resulting factor')
        ret = Factor.normalize(product)
        printFactors([ret])
        return ret

def printFactors(factorlist):
    print('*****Factors*****')
    for factor in factorlist:
        print('     factor(' + ','.join(factor.varlist) + ') = ', end=''),
        print(factor.flist)

        
def main():
    #Construct a factor for each conditional probability
    print('Step 1: Construct factors:')
    #lecture example
    #flist1 = np.array([0.0003, 0.9997])
    #f1 = Factor(['E'], [['t', 'f']], flist1)
    #flist2 = np.array([0.0001, 0.9999])
    #f2 = Factor(['B'], [['t', 'f']], flist2)
    #flist3 = np.array([[0.9, 0.0002], [0.1, 0.9998]])
    #f3 = Factor(['R', 'E'], [['t', 'f'], ['t', 'f']], flist3)
    #flist4 = np.array([[[0.96, 0.95], [0.2, 0.01]], [[0.04, 0.05], [0.08, 0.99]]])
    #f4 = Factor(['A', 'B', 'E'], [['t', 'f'], ['t', 'f'], ['t', 'f']], flist4)
    #flist5 = np.array([[0.8, 0.4], [0.4, 0.6]])
    #f5 = Factor(['W', 'A'], [['t', 'f'], ['t', 'f']], flist5)
    #flist6 = np.array([[0.4, 0.04], [0.6, 0.96]])
    #f6 = Factor(['G', 'A'], [['t', 'f'], ['t', 'f']], flist6)

    #factorlist = [f1, f2, f3, f4, f5, f6]
    #querylist = ['B']
    #evidencelist = [('W', 't'), ('G', 't')]
    #hiddenvarlist = ['R', 'E', 'A']

    #Q2
    #flist1 = np.array([0.05, 0.95])
    #f1 = Factor(['S'], [['t', 'f']], flist1)
    #flist2 = np.array([1/28, 1 - 1/28])
    #f2 = Factor(['M'], [['t', 'f']], flist2)
    #flist3 = np.array([0.3, 0.7])
    #f3 = Factor(['NA'], [['t', 'f']], flist3)
    #flist4 = np.array([[0.6, 0.1], [0.4, 0.9]])
    #f4 = Factor(['B', 'S'], [['t', 'f'], ['t', 'f']], flist4)
    #flist5 = np.array([[[0.8, 0.4], [0.5, 0.]], [[0.2, 0.6], [0.5, 1.]]])
    #f5 = Factor(['NH', 'M', 'NA'], [['t', 'f'], ['t', 'f'], ['t', 'f']], flist5)
    #flist6 = np.array([[[[0.99, 0.9 ], [0.75, 0.5 ]], [[0.65, 0.4 ], [0.2 , 0.  ]]],
    #                   [[[0.01, 0.1 ], [0.25, 0.5 ]], [[0.35, 0.6 ], [0.8 , 1.  ]]]])
    #f6 = Factor(['FH', 'S', 'M', 'NH'], [['t', 'f'], ['t', 'f'], ['t', 'f'], ['t', 'f']], flist6)

    #Q2b
    #factorlist = [f1, f2, f3, f4, f5, f6]
    #querylist = ['FH']
    #evidencelist = [#]
    #hiddenvarlist = ['B', 'M', 'NA', 'NH', 'S']

    #Q2c
    #factorlist = [f1, f2, f3, f4, f5, f6]
    #querylist = ['S']
    #evidencelist = [('FH', 't'), ('M', 't')]
    #hiddenvarlist = ['B', 'NA', 'NH']
    
    #Q2d
    #factorlist = [f1, f2, f3, f4, f5, f6]
    #querylist = ['S']
    #evidencelist = [('FH', 't'), ('M', 't'), ('B', 't')]
    #hiddenvarlist = ['NA', 'NH']

    #Q2e
    #factorlist = [f1, f2, f3, f4, f5, f6]
    #querylist = ['S']
    #evidencelist = [('FH', 't'), ('M', 't'), ('B', 't'), ('NA', 't')]
    #hiddenvarlist = ['NH']

    #Q3
    flist1 = np.array([0.1, 0.9])
    f1 = Factor(['B'], [['t', 'f']], flist1)
    flist2 = np.array([0.05, 0.95])
    f2 = Factor(['E'], [['t', 'f']], flist2)
    flist3 = np.array([[[0.95, 0.1], [0.9, 0.05]], [[0.05, 0.9], [0.1, 0.95]]])
    f3 = Factor(['A', 'E', 'B'], [['t', 'f'], ['t', 'f'], ['t', 'f']], flist3)
    flist4 = np.array([[0.8, 0.4], [0.2, 0.6]])
    f4 = Factor(['W', 'A'], [['t', 'f'], ['t', 'f']], flist4)
    flist5 = np.array([[0.4, 0.05], [0.6, 0.95]])
    f5 = Factor(['G', 'A'], [['t', 'f'], ['t', 'f']], flist5)

    #Q3.1
    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['G']
    #evidencelist = [('W', 't')]
    #hiddenvarlist = ['A', 'B', 'E']

    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['G']
    #evidencelist = [('W', 'f')]
    #hiddenvarlist = ['A', 'B', 'E']

    #Q3.2
    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['B']
    #evidencelist = [('W', 't'), ('G', 't'), ('A', 't')]
    #hiddenvarlist = ['E']

    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['B']
    #evidencelist = [('A', 't')]
    #hiddenvarlist = ['W', 'G', 'E']

    #Q3.3
    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['B']
    #videncelist = [('W', 't')]
    #hiddenvarlist = ['G', 'A', 'E']

    #Q3.4
    #factorlist = [f1, f2, f3, f4, f5]
    #querylist = ['E']
    #evidencelist = [('A', 't'), ('B', 't')]
    #hiddenvarlist = ['W', 'G']

    factorlist = [f1, f2, f3, f4, f5]
    querylist = ['E']
    evidencelist = [('A', 't')]
    hiddenvarlist = ['W', 'G', 'B']

    printFactors(factorlist)
    final_result = Factor.inference(factorlist, querylist, evidencelist, hiddenvarlist)
    return final_result

if __name__ == '__main__':
    main()