import numpy as np
from anastruct.fem.system import SystemElements

class GeneticAlgorithm:
    popLength = 0
    mutationRate = 0
    keep = 0
    newIndiv = 0
    (grid) = ([0, 0])
    force = 0
    EA = 0
    EI = 0

    loc = []
    comb = []
    population = []

    # Initialization of the algorithm
    def __init__(self, popLength, mutationRate, keep, newIndiv, requirements):
        self.popLength = popLength
        self.mutationRate = mutationRate
        self.keep = keep
        self.newIndiv = newIndiv
        (self.grid) = requirements[0]
        self.force = requirements[1]
        self.EA = requirements[2]
        self.EI = requirements[3]
        

        xLen = self.grid[0]
        zLen = self.grid[1]
        
        # Generating loc array
        for k in range(zLen + 1):
            for i in range(xLen + 1):
                self.loc.append([i, k])
  
        # Generating comb array
        for k in range(zLen):
            self.comb.append([self.loc.index([0, k]), self.loc.index([1, k])])
            self.comb.append([self.loc.index([0, k]), self.loc.index([0, k+1])])
            self.comb.append([self.loc.index([0, k]), self.loc.index([1, k+1])])
            for i in range(1, xLen):
                self.comb.append([self.loc.index([i, k]), self.loc.index([i+1, k])])
                self.comb.append([self.loc.index([i, k]), self.loc.index([i-1, k+1])])
                self.comb.append([self.loc.index([i, k]), self.loc.index([i, k+1])])
                self.comb.append([self.loc.index([i, k]), self.loc.index([i+1, k+1])])
            self.comb.append([self.loc.index([xLen, k]), self.loc.index([xLen-1, k+1])])
            self.comb.append([self.loc.index([xLen, k]), self.loc.index([xLen, k+1])])
        for k in range(xLen):
            self.comb.append([self.loc.index([k, zLen]), self.loc.index([k+1, zLen])])   
        
        # Creating initial population
        k = 0
        while k < self.popLength:
            try:    
                DNA = self.DNA()
                indiv = self.createTruss(DNA)
                self.population.append(indiv)
                k += 1
            except:
                pass
        
        print('Genetic Algorithm ready to go !')
        print('   ')

    # Generates the shape of one struss structure
    def DNA(self):
        xLen = self.grid[0]
        zLen = self.grid[1]
        size = 4*xLen*zLen + xLen + zLen
        DNA_Created = np.random.randint(0, 2, size=size)

        return DNA_Created

    # Creates the truss structure Python object
    def createTruss(self, DNA):
        ts = SystemElements(EA=15000, EI=5000)
        for k in range(len(DNA)):
            if DNA[k] == 1:
                n1 = self.comb[k][0]
                n2 = self.comb[k][1]
                l1 = self.loc[n1]
                l2 = self.loc[n2]
                ts.add_element(location=[l1, l2])
           
        nodeSuppId1 = ts.find_node_id([0, 0])
        nodeSuppId2 = ts.find_node_id([self.grid[0], 0])
        nodeLoadId = ts.find_node_id([self.grid[0]//2, 0])
        
        ts.add_support_hinged(nodeSuppId1)
        ts.add_support_hinged(nodeSuppId2)
        ts.point_load(nodeLoadId, Fz=self.force)

        ts.solve()
        
        nodes = [nodeSuppId1, nodeSuppId2, nodeLoadId]
        cost = self.costFunction(ts, DNA, nodeLoadId)
        return [ts, DNA, nodes, cost]

    # Returns how well a solution matches with the requirements
    def costFunction(self, ts, DNA, nodeLoadId):
        displacement = np.abs(ts.get_node_displacements(nodeLoadId)["uy"])
        segments = 0
        for k in range(len(DNA)):
            if DNA[k] == 1:
                segments += 1

        C1 = 100
        C2 = 0.02
        cost = C1*displacement + C2*segments
        return cost
        
    def start(self, nb_iterations):
        print("Lancement de l'algorithme ...")
        print("   ")
        self.population[0][0].show_structure()
        for k in range(nb_iterations):
            print('Loop : ' + str(k))
            self.mutation()
            self.sort()
            self.crossOver()
            self.sort()
            if k%10 == 0:
                self.stats()
                self.showTruss(self.population[0][1])
                
        print('  ')
        print('End')

    # Sorts all solutions by how much they cost in ascending order
    def sort(self):
        self.population = sorted(self.population, key = lambda col: col[3])

    # Applies mutations to a solutions depending on the mutation rate
    def mutation(self):
        for k in range(len(self.population)):
            p = np.random.randint(100)
            if p <= self.mutationRate:
                try:
                    mDNA = np.copy(self.population[k][1])
                    i = np.random.randint(len(mDNA))
                    if mDNA[i] == 1:
                        mDNA[i] = 0
                    else:
                        mDNA[i] = 1
                    mIndiv = self.createTruss(mDNA)

                    del(self.population[k])
                    self.population.append(mIndiv)
                except:
                    pass
        
    # Creates a solution using two others one
    def crossOver(self):
        self.population = self.population[:self.keep]

        while len(self.population) < self.popLength - self.newIndiv:
            try:
                p1 = np.random.randint(1, len(self.population))
                p2 = p1
                while p2 == p1:
                    p2 = np.random.randint(1, len(self.population))
                    
                DNA_P1 = self.population[p1][1]
                DNA_P2 = self.population[p2][1]
                c = np.random.randint(1, len(DNA_P1))
                DNA_Child = np.array(list(DNA_P1[:c]) + list(DNA_P2[c:]))

                indiv = self.createTruss(DNA_Child)
                self.population.append(indiv)
            except:
                pass
            
        while len(self.population) <= self.popLength:
            try:    
                DNA = self.DNA()
                indiv = self.createTruss(DNA)
                self.population.append(indiv)
            except:
                pass

    # What is actually displayed on the user screen
    def stats(self):
        print('  ')
        print('-----')
        print('  ')
        for k in range(len(self.population)):
            print("Cost Result :", self.population[k][3], "- DNA :", self.population[k][1])

    # Displays the shape of a generated truss structure
    def showTruss(self, DNA):
        indiv = self.createTruss(DNA)
        indiv[0].show_structure()
        indiv[0].show_displacement()
        

# Genetic Algorithm parameters
popLength = 100
mutationRate = 7
keep = 70
newIndiv = 10
nb_iteration = 200

# Truss structures parameters
grid = [6, 1]
force = -100 #*10^3
EA = 15000 # Standard axial stiffness of elements
EI = 5000 # Standard bending stiffness of elements
requirements = (grid, force, EA, EI)


print('Lanching algorithm !')

geneAlgo = GeneticAlgorithm(popLength, mutationRate, keep, newIndiv, requirements)
geneAlgo.start(nb_iteration)

#geneAlgo.showTruss([1 ,1 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,1 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,1 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,0 ,1 ,0,0 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,1 ,0 ,0 ,1 ,0 ,0 ,1 ,1 ,1 ,1,0])
