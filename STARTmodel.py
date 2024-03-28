import pandas as pd
from gurobipy import quicksum, Model, GRB

class START:
    def __init__(self, excel_file, pWI, pWC, pWA, pWO, pWG, pInfeas, pWGC, pWGA, pWGG, pGoalc, pGoala, pGoalg, vBetaini, pNpeople, pNperiods, pNhealthp, pMaxgrade, pIdc, pIda, pIdg):
        super(START).__init__()
        self.excel_file = excel_file
        self.data = pd.read_excel(excel_file, sheet_name='Data', header=None)
        self.pNperiods = self.data.loc[0,1]
        self.pNhealthp = self.data.loc[1,1]
        self.pNpeople = self.data.loc[2,1]
        self.pNcharter = self.data.loc[3,1]
        self.pDiscount = self.data.loc[4,1]
        self.pKpeople = self.data.loc[5,1]
        self.pMaxperiods = self.data.loc[6,1]
        self.pMinperiods = self.data.loc[7,1]

        self.prices = pd.read_excel(excel_file, sheet_name='Prices')
        self.pCostOut = self.prices.loc[:,'Outward'].to_numpy()
        self.pCostRet = self.prices.loc[:,'Return'].to_numpy()

        self.charter = pd.read_excel(excel_file, sheet_name='Charter')
        self.pCostChar = self.charter.loc[0:self.pNperiods-1,'Chartered 1':'Chartered 2'].to_numpy()
        self.pMinCapCh = self.charter.loc[0:self.pNcharter-1,'Min Cap'].to_numpy()
        self.pMaxCapCh = self.charter.loc[0:self.pNcharter-1,'Max Cap'].to_numpy()

        self.demand = pd.read_excel(excel_file, sheet_name='Demand')
        self.pHealthdem = self.demand.loc[:,1:].to_numpy()

        self.healthprofiles = pd.read_excel(excel_file, sheet_name='HealthProfiles', header=None)
        self.pNameprofiles = self.healthprofiles.loc[0,1:].to_numpy()
        self.pNameabbprofiles = self.healthprofiles.loc[1,1:].to_numpy()
        self.pProfiles = self.healthprofiles.loc[2:,1:].to_numpy()

        self.availability = pd.read_excel(excel_file, sheet_name='Availability')
        self.pAvailability = self.availability.loc[:,1:].to_numpy()

        self.grades = pd.read_excel(excel_file, sheet_name='Grades')
        self.pGrades = self.grades.loc[0:self.pNpeople-1,'Grade'].to_numpy()

        #self.weights = pd.read_excel(excel_file, sheet_name='Weights', header=None)
        #self.pW1 = self.weights.loc[0,1]
        #self.pW2 = self.weights.loc[1,1]
        #self.pW3 = self.weights.loc[2,1]

        #self.pWW1 = self.weights.loc[4,1]
        #self.pWW2 = self.weights.loc[5,1]
        #self.pWW3 = self.weights.loc[6,1]
        #self.pWW4 = self.weights.loc[7,1]

        self.vAlphaout = {}
        self.vAlpharet = {}
        self.vBeta = {}
        self.vGamma = {}
        self.vDeltaout = {}
        self.vDeltaret = {}
        self.vMu = {}
        self.vXstand = {}
        self.vXdisc = {}
        self.vYstand = {}
        self.vYdisc = {}
        self.vZout = {}
        self.vZret = {}
        self.vUmenos = {}
        self.vUmas = {}
        self.vFGrade = {}
        self.vFAvo = {}
        self.vFAvr = {}
        self.vFAv = {}
        self.vMeanavper = {}
        self.vBetaini = vBetaini

        self.pNperiods = pNperiods
        self.pNhealthp = pNhealthp
        self.pNpeople = pNpeople

        self.pInfeas = pInfeas
        self.pWI = pWI
        self.pWC = pWC
        self.pWA = pWA
        self.pWG = pWG
        self.pWO = pWO
        self.pWGC = pWGC
        self.pWGA = pWGA
        self.pWGG = pWGG
        self.pGoalc = pGoalc
        self.pGoala = pGoala
        self.pGoalg = pGoalg
        self.pMaxgrade = pMaxgrade
        self.pIdc = pIdc
        self.pIda = pIda
        self.pIdg = pIdg

        self.pMaxgrade = max(self.pGrades)


    # CREATE MODEL
    def create_model(self):
        self.model = Model("START")
        self.model.setParam("TimeLimit", 7200)
        #self.model.setParam('MIPGap', 0.01)
        self.model.setParam('OutputFlag', 0)

        return self.model

    # VARIABLES DEFINITION
    def create_variables(self): 
        # Binary for person i taking an outward flight in time period t
        for i in range(0, self.pNpeople):
             for t in range(0, self.pNperiods):
                self.vAlphaout[i,t] = self.model.addVar(name='vAlphaout_%s_%s' % (i, t), vtype=GRB.BINARY)
        
        # Binary for person i taking a return flight in time period t
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.vAlpharet[i,t] = self.model.addVar(name='vAlpharet_%s_%s' % (i, t), vtype=GRB.BINARY)
        
        # Binary for person i attending with profile j in time period t
        for i in range(0, self.pNpeople):
            for j in range(0, self.pNhealthp):
                for t in range(0, self.pNperiods):
                    self.vBeta[i, j, t] = self.model.addVar(ub=min(self.pAvailability[i,t],self.pProfiles[i,j]), name='vBeta_%s_%s_%s' % (i, j, t), vtype=GRB.BINARY)
                    # Initial solution
                    if self.vBetaini:
                        self.vBeta[i,j,t].Start = self.vBetaini[i,j,t]

        # Binary for chartered flight type 1 in period 
        for t in range(0, self.pNperiods):
            for l in range(0, self.pNcharter):
                self.vGamma[t,l] = self.model.addVar(name='vGamma_%s_%s' % (t,l), vtype=GRB.BINARY)
        
        # Binary auxiliary for applying or not the discount in outward flights
        for t in range(0, self.pNperiods):
            self.vDeltaout[t] = self.model.addVar(name='vDeltaout_%s' % (t), vtype=GRB.BINARY)

        # Binary auxiliary for applying or not the discount in outward flights
        for t in range(0, self.pNperiods):
            self.vDeltaret[t] = self.model.addVar(name='vDeltaret_%s' % (t), vtype=GRB.BINARY)
        
        # Binary variable if person i works with role j
        for i in range(0, self.pNpeople):
            for j in range(0, self.pNhealthp):
                self.vMu[i,j] = self.model.addVar(ub=self.pProfiles[i,j], name='vMu_%s_%s' %(i,j), vtype=GRB.BINARY)

        # Number of people taking an outward flight with standard fare in time period t
        for t in range(0, self.pNperiods):
            self.vXstand[t] = self.model.addVar(lb=0.0, name='vXstand_%s' % (t), vtype=GRB.INTEGER)

        # Number of people taking an outward flight with discounted fare in time period t
        for t in range(0, self.pNperiods):
            self.vXdisc[t] = self.model.addVar(lb=0.0, name='vXdisc_%s' % (t), vtype=GRB.INTEGER)
        
        # Number of people taking a return flight with standard fare in time period t
        for t in range(0, self.pNperiods):
            self.vYstand[t] = self.model.addVar(lb=0.0, name='vYstand_%s' % (t), vtype=GRB.INTEGER)
        
        # Number of people taking a return flight with discounted fare in time period t
        for t in range(0, self.pNperiods):
            self.vYdisc[t] = self.model.addVar(lb=0.0, name='vYdisc_%s' % (t), vtype=GRB.INTEGER)
        
        # Number of people taking a chartered flight outwards in period t
        for t in range(0, self.pNperiods):
            self.vZout[t] = self.model.addVar(lb=0.0, name='vZout_%s' % (t), vtype=GRB.INTEGER)
        
        # Number of people taking a chartered flight outwards in period t
        for t in range(0, self.pNperiods):
            self.vZret[t] = self.model.addVar(lb=0.0, name='vZret_%s' % (t), vtype=GRB.INTEGER)
        
        # Slack variable
        for j in range(0, self.pNhealthp):
            for t in range(0, self.pNperiods):
                self.vUmenos[j,t] = self.model.addVar(lb=0.0, name='vUmenos_%s_%s' % (j,t), vtype=GRB.INTEGER)
        
        # Surplus variable
        for j in range(0, self.pNhealthp):
            for t in range(0, self.pNperiods):
                self.vUmas[j,t] = self.model.addVar(lb=0.0, name='vUmas_%s_%s' % (j,t), vtype=GRB.INTEGER)

        # Variables for Fortet linealisations
        # Grades
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.vFGrade[i,t] = self.model.addVar(lb=0.0, ub=10.0, name='vFGrade_%s_%s' % (i,t), vtype=GRB.CONTINUOUS)

        # Availability outward
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.vFAvo[i,t] = self.model.addVar(lb=0.0, ub=2.0, name='vFAvo_%s_%s' % (i,t), vtype=GRB.CONTINUOUS)

        # Availability return
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.vFAvr[i,t] = self.model.addVar(lb=0.0, ub=2.0, name='vFAvr_%s_%s' % (i,t), vtype=GRB.CONTINUOUS)

        # Availability global
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.vFAv[i,t] = self.model.addVar(lb=0.0, ub=2.0, name='vFAv_%s_%s' % (i,t), vtype=GRB.CONTINUOUS)

        # Mean grade
        self.vMeangrade = self.model.addVar(lb=0.0, ub=10.0, name='vMeangrade', vtype=GRB.CONTINUOUS)

        # Mean availability per person
        for i in range(0, self.pNpeople):
            self.vMeanavper[i] = self.model.addVar(lb=0.0, ub=2.0, name='vMeanavper_%s' %i, vtype=GRB.CONTINUOUS)

        # Mean availability
        self.vMeanav = self.model.addVar(lb=0.0, ub=2.0, name='vMeanavail', vtype=GRB.CONTINUOUS)

        # Variable for infeasibility (objective function)
        self.vInfeas = self.model.addVar(lb=0.0, name='vInfeas', vtype=GRB.CONTINUOUS)

        # Variable for cost (objective function)
        self.vCost = self.model.addVar(lb=0.0, name='vCost', vtype=GRB.CONTINUOUS)

        # Variable for availability (objective function)
        #self.vAvailab = self.model.addVar(lb=0.0, name='vAvailab', vtype=GRB.CONTINUOUS)

        # Variable for Grades
        #self.vGrades = self.model.addVar(lb=0.0, name='vGrades', vtype=GRB.CONTINUOUS)

        # Variable for one role
        self.vOnerole = self.model.addVar(lb=0.0, name='vOnerole', vtype=GRB.CONTINUOUS)

        # Variable for number of people travelling
        self.vTotpeop = self.model.addVar(lb=0.0, name='vTotPeop', vtype=GRB.INTEGER)

        # Variables for deviation of goal for cost
        self.vDevc1 = self.model.addVar(lb=0.0, name='vDevc1', vtype=GRB.CONTINUOUS)
        self.vDevc2 = self.model.addVar(lb=0.0, name='vDevc2', vtype=GRB.CONTINUOUS)

        # Variables for deviation of goal for availability
        self.vDeva1 = self.model.addVar(lb=0.0, name='vDeva1', vtype=GRB.CONTINUOUS)
        self.vDeva2 = self.model.addVar(lb=0.0, name='vDeva2', vtype=GRB.CONTINUOUS)

        # Variables for deviation of goal for grades
        self.vDevg1 = self.model.addVar(lb=0.0, name='vDevg1', vtype=GRB.CONTINUOUS)
        self.vDevg2 = self.model.addVar(lb=0.0, name='vDevg2', vtype=GRB.CONTINUOUS)

    # OBJECTIVE FUNCTION
    def create_objective_function(self):
        # Objective function definition with parameters for any of the objectives
        self.model.setObjective(self.pWI*self.vInfeas +
                                #self.pWC*self.vCost + self.pWA*self.vMeanav + self.pWG*self.vMeangrade +
                                self.pWC * (self.vCost - self.pIdc) + self.pWA * (self.pIda - self.vMeanav) + self.pWG * (self.pIdg - self.vMeangrade) +
                                self.pWGC*self.vDevc2 + self.pWGA*self.vDeva1 + self.pWGG*self.vDevg1 +
                                self.pWO*self.vOnerole,
                                sense=GRB.MINIMIZE)

    # CONSTRAINTS
    def create_constrains(self):

        # Health profiles must be covered
        for j in range(0, self.pNhealthp):
            for t in range(0, self.pNperiods-1):
                self.model.addConstr(quicksum(self.pProfiles[i,j]*self.vBeta[i,j,t] for i in range(0, self.pNpeople)) - self.vUmenos[j,t] + self.vUmas[j,t] == self.pHealthdem[j,t], 'Profiles_%s_%s' %(j,t))

        # Minimum number of periods
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum((t+1)*self.vAlpharet[i,t] for t in range(1, self.pNperiods)) - quicksum((t+1)*self.vAlphaout[i,t] for t in range(0,self.pNperiods-1)) >= self.pMinperiods * quicksum(self.vAlphaout[i, t] for t in range(0, self.pNperiods)), 'Minperiodsalpha_%s' %i)
        
        # Maximum number of periods
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum((t+1)*self.vAlpharet[i,t] for t in range(1, self.pNperiods)) - quicksum((t+1)*self.vAlphaout[i,t] for t in range(0,self.pNperiods-1)) <= self.pMaxperiods * quicksum(self.vAlphaout[i, t] for t in range(0, self.pNperiods)), 'Maxperiodsalpha_%s' %i)
        
        # Maximum number of periods with beta
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum(quicksum(self.vBeta[i,j,t] for j in range(0, self.pNhealthp)) for t in range(0, self.pNperiods)) <= self.pMaxperiods, 'Maxperiodsbeta_%s' %i)
        
        # Controlling the outward and return flights for each person
        for i in range(0, self.pNpeople):
            for t in range (0, self.pNperiods):
                if t > 0:
                    self.model.addConstr(quicksum(self.vBeta[i,j,t] for j in range(0, self.pNhealthp)) - quicksum(self.vBeta[i,j,t-1] for j in range(0, self.pNhealthp)) == self.vAlphaout[i,t] - self.vAlpharet[i,t], 'Flights_%s_%s' %(i,t))
        
        # Controlling the outward and return flights for each person in first period
        for i in range(0, self.pNpeople):
            for t in range (0,1):
                self.model.addConstr(quicksum(self.vBeta[i,j,t] for j in range(0, self.pNhealthp)) == self.vAlphaout[i,t], 'Flights1_%s_%s' %(i,t))
        
        # One health profile for person at most for each time period
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.model.addConstr(quicksum(self.vBeta[i, j, t] for j in range(0, self.pNhealthp)) <= 1, 'Oneprofile_%s_%s' % (i, t))
        
        # Each person can fly once at most outward
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum(self.vAlphaout[i, t] for t in range(0, self.pNperiods-1)) <= 1, 'Onceout_%s' % i)

        # Each person can fly once at most return
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum(self.vAlpharet[i, t] for t in range(1, self.pNperiods)) <= 1, 'Onceret_%s' % i)
        
        # Number of outward are counted
        for t in range(0, self.pNperiods-1):
            self.model.addConstr(self.vXstand[t] + self.vXdisc[t] + self.vZout[t] == quicksum(self.vAlphaout[i, t] for i in range(0, self.pNpeople)), 'Countout_%s' % t)

        # Number of return flights are counted
        for t in range(1, self.pNperiods):
            self.model.addConstr(self.vYstand[t] + self.vYdisc[t] + self.vZret[t] == quicksum(self.vAlpharet[i, t] for i in range(0, self.pNpeople)), 'Countret_%s' % t)
        
        # Outward standard fare definition
        for t in range(0, self.pNperiods-1):
            self.model.addConstr(self.vXstand[t] <= (self.pKpeople-1)*self.vDeltaout[t], 'StandOut_%s' % t)
        
        # Outward discounted fare definition 1
        for t in range(0, self.pNperiods-1):
            self.model.addConstr(self.pKpeople*(1-self.vDeltaout[t]) <= self.vXdisc[t], 'DiscOut1_%s' % t)

        # Outward discounted fare definition 2
        for t in range(0, self.pNperiods-1):
            self.model.addConstr(self.vXdisc[t] <= self.pNpeople*(1-self.vDeltaout[t]), 'DiscOut2_%s' % t)
        
        # Return standard fare definition
        for t in range(1, self.pNperiods):
            self.model.addConstr(self.vYstand[t] <= (self.pKpeople-1)*self.vDeltaret[t], 'StandRet_%s' % t)
        
        # Return discounted fare definition 1
        for t in range(1, self.pNperiods):
            self.model.addConstr(self.pKpeople*(1-self.vDeltaret[t]) <= self.vYdisc[t], 'DiscRet1_%s' % t)
        
        # Return discounted fare definition 2
        for t in range(1, self.pNperiods):
            self.model.addConstr(self.vYdisc[t] <= self.pNpeople*(1-self.vDeltaret[t]), 'DiscRet2_%s' % t)
        
        # Only one type of chartered flight is hired at most
        for t in range(0, self.pNperiods):
            self.model.addConstr(quicksum(self.vGamma[t,l] for l in range(0,self.pNcharter)) <= 1, 'Onechtype_%s' % t)

        # A chartered flight is hired on first and last periods
        self.model.addConstr(quicksum(self.vGamma[0, l] for l in range(0, self.pNcharter)) == 1, 'Chartered1')
        self.model.addConstr(quicksum(self.vGamma[self.pNperiods-1, l] for l in range(0, self.pNcharter)) == 1, 'Chartered2')
        
        # Satisfying minimum capacity of chartered in outward flights
        for t in range(0, self.pNperiods - 1):
            self.model.addConstr(quicksum(self.pMinCapCh[l] * self.vGamma[t,l] for l in range(0,self.pNcharter)) <= self.vZout[t], 'Mincapout_%s' % t)
        
       # Satisfying maximum capacity of chartered in outward flights
        for t in range(0, self.pNperiods - 1):
            self.model.addConstr(self.vZout[t] <= quicksum(self.pMaxCapCh[l] * self.vGamma[t,l] for l in range(0, self.pNcharter)), 'Mincapout_%s' % t)
        
         # Satisfying minimum capacity of chartered in return flights
        for t in range(1, self.pNperiods):
            self.model.addConstr(quicksum(self.pMinCapCh[l] * self.vGamma[t, l] for l in range(0, self.pNcharter)) <= self.vZret[t], 'Mincapout_%s' % t)
        
        # Satisfying maximum capacity of chartered in return flights
        for t in range(1, self.pNperiods):
            self.model.addConstr(self.vZret[t] <= quicksum(self.pMaxCapCh[l] * self.vGamma[t, l] for l in range(0, self.pNcharter)), 'Mincapout_%s' % t)

        # Number of roles that a person plays 1
        for i in range(0, self.pNpeople):
            for j in range(0, self.pNhealthp):
                self.model.addConstr(quicksum(self.vBeta[i,j,t] for t in range(0,self.pNperiods-1)) <= self.pMaxperiods * self.vMu[i,j], 'Roles1_%s_%s' %(i,j))

        # Number of roles that a person plays 2
        for i in range(0, self.pNpeople):
            for j in range(0, self.pNhealthp):
                self.model.addConstr(quicksum(self.vBeta[i,j,t] for t in range(0,self.pNperiods-1)) >= self.vMu[i,j], 'Roles1_%s_%s' %(i,j))

        # No infeasibility
        self.model.addConstr(quicksum(quicksum(self.vUmas[j, t] for j in range(0, self.pNhealthp)) for t in range(0, self.pNperiods - 1)) <= self.pInfeas, 'NoInfeasibility')

        # FORTET LINEALISATIONS
        # Fortet linealisation for mean grades
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.model.addConstr(self.vFGrade[i, t] <= self.pMaxgrade*self.vAlphaout[i,t], 'F1Grades')
                self.model.addConstr(self.vFGrade[i, t] <= self.vMeangrade, 'F2Grades')
                self.model.addConstr(self.vMeangrade - self.vFGrade[i, t] <= self.pMaxgrade*(1-self.vAlphaout[i,t]), 'F3Grades')

        # Fortet linealisation for mean availability per person (outward)
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.model.addConstr(self.vFAvo[i, t] <= 2*self.vAlphaout[i,t], 'F1AvPerOut')
                self.model.addConstr(self.vFAvo[i, t] <= self.vMeanavper[i], 'F2AvPerOut')
                self.model.addConstr(self.vMeanavper[i] - self.vFAvo[i, t] <= 2*(1-self.vAlphaout[i,t]), 'F3AvPerOut')

        # Fortet linealisation for mean availability per person (outward)
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.model.addConstr(self.vFAvr[i, t] <= 2 * self.vAlpharet[i, t], 'F1AvPerRet')
                self.model.addConstr(self.vFAvr[i, t] <= self.vMeanavper[i], 'F2AvPerRet')
                self.model.addConstr(self.vMeanavper[i] - self.vFAvr[i, t] <= 2 * (1 - self.vAlpharet[i, t]), 'F3AvPerRet')

        # Fortet linealisation for mean availability (global)
        for i in range(0, self.pNpeople):
            for t in range(0, self.pNperiods):
                self.model.addConstr(self.vFAv[i, t] <= 2 * self.vAlphaout[i, t], 'F1Av')
                self.model.addConstr(self.vFAv[i, t] <= self.vMeanav, 'F2Av')
                self.model.addConstr(self.vMeanav - self.vFAv[i, t] <= 2 * (1 - self.vAlphaout[i, t]), 'F3Av')

        # Original equations for means
        # Grades
        self.model.addConstr(quicksum(quicksum(self.vFGrade[i,t] for i in range(0, self.pNpeople)) for t in range(0, self.pNperiods)) ==
                             quicksum(quicksum(self.pGrades[i]*self.vAlphaout[i,t] for i in range(0, self.pNpeople)) for t in range(0, self.pNperiods)) , 'Meangrades')

        # Availability per person
        for i in range(0, self.pNpeople):
            self.model.addConstr(quicksum((t*self.vFAvr[i,t] - t*self.vFAvo[i,t]) for t in range(0, self.pNperiods)) - self.vMeanavper[i]
                                 == quicksum(quicksum(self.pAvailability[i, t]*self.vBeta[i,j,t] for j in range(0, self.pNhealthp)) for t in range(0,self.pNperiods)), 'Meanavperson')

        # Global availability
        self.model.addConstr(quicksum(quicksum(self.vFAv[i, t] for i in range(0, self.pNpeople)) for t in range(0, self.pNperiods)) ==
                             quicksum(self.vMeanavper[i] for i in range(0, self.pNpeople)), 'Meanav')

        # OBJECTIVES DEFINITION AS VARIABLES
        # Infeasibility
        self.model.addConstr(self.vInfeas == quicksum(quicksum(self.vUmas[j, t] for j in range(0, self.pNhealthp)) for t in range(0, self.pNperiods - 1)), 'Infeasibility')

        # Cost
        self.model.addConstr(self.vCost == quicksum(quicksum(self.pCostChar[t,l] * self.vGamma[t,l] for t in range(0,self.pNperiods)) for l in range(0,self.pNcharter)) +
                                           (quicksum(self.pCostOut[t] * self.vXstand[t] + (self.pCostOut[t] - self.pDiscount) * self.vXdisc[t] for t in range(0,self.pNperiods-1)) +
                                            quicksum(self.pCostRet[t] * self.vYstand[t] + (self.pCostRet[t] - self.pDiscount) * self.vYdisc[t] for t in range(1,self.pNperiods))), 'Cost')

        # Availability
        #self.model.addConstr(self.vAvailab == quicksum(quicksum(quicksum(self.pAvailability[i, t] * self.vBeta[i, j, t]
        #                                 for i in range(0, self.pNpeople)) for j in range(0, self.pNhealthp)) for t in range(0, self.pNperiods)), 'Availability')

        # Grades
        #self.model.addConstr(self.vGrades == quicksum(quicksum(self.pGrades[i]*self.vAlphaout[i, t] for i in range(0, self.pNpeople)) for t in range(0, self.pNperiods)), 'TotGrade')

        # One role
        self.model.addConstr(self.vOnerole == quicksum(quicksum(self.vMu[i, j] for i in range(0, self.pNpeople)) for j in range(0, self.pNhealthp)), 'Onerole')

        # Total number of people attending the emergency
        self.model.addConstr(self.vTotpeop == quicksum(self.vXstand[t] + self.vXdisc[t] + self.vZout[t] for t in range(0, self.pNperiods)), 'Total people')

        # CONSTRAINTS FOR GOAL PROGRAMMING
        # Goal cost:
        self.model.addConstr(self.vCost + self.vDevc1 - self.vDevc2 == self.pGoalc)

        # Goal availability
        self.model.addConstr(self.vMeanav + self.vDeva1 - self.vDeva2 == self.pGoala)

        # Goal grades
        self.model.addConstr(self.vMeangrade + self.vDevg1 - self.vDevg2 == self.pGoalg)
    
    def solve(self):
        self.model.optimize()
        return self.model





