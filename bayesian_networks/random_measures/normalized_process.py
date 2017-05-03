'''
Created on May 3, 2017

@author: cesar
'''
import numpy as np

def ChineseRestaurantProcess(numberOfCostumers,lambda_measure):
    """
    Chinese restaurant process
    """
    costumer_seats = []
    theta = lambda_measure.generate_points(1)[0]
    alpha = lambda_measure.interval_size
    Thetas = [theta]
    costumer_seats.append(theta)
    
    p = [1./(alpha + 1), alpha/(alpha + 1)]
    numberOfSeatedCostumers = [1.]
    numberOfTables = 1
    for i in range(numberOfCostumers-1):
        p = np.concatenate([np.array(numberOfSeatedCostumers),[alpha]])/(alpha + sum(numberOfSeatedCostumers))
        selectedTable = np.random.choice(np.arange(numberOfTables+1),p=p)
        if selectedTable == numberOfTables:
            #NewTableSelected
            numberOfTables += 1
            theta = lambda_measure.generate_points(1)[0]
            numberOfSeatedCostumers.append(1.)
            Thetas.append(theta)
            costumer_seats.append(theta)
        else:
            numberOfSeatedCostumers[selectedTable] = numberOfSeatedCostumers[selectedTable] + 1.
            costumer_seats.append(Thetas[selectedTable])
            
    return (costumer_seats,Thetas,numberOfSeatedCostumers)

def ExtendedChineseRestaurantProcess(self,numberOfCostumers,lambda_measure,used_tables,already_seated):
    """
    Chinese restaurant process
    """
    theta = np.random.choice(self.inhomogeneousPoisson())
    Thetas = [theta]
    p = [1./(self.alpha + 1), self.alpha/(self.alpha + 1)]
    numberOfSeatedCostumers = [1.]
    numberOfTables = 1
    for i in range(numberOfCostumers-1):
        p = np.concatenate([np.array(numberOfSeatedCostumers),[self.alpha]])/(self.alpha + sum(numberOfSeatedCostumers))
        selectedTable = np.random.choice(np.arange(numberOfTables+1),p=p)
        if selectedTable == numberOfTables:
            #NewTableSelected
            numberOfTables += 1
            theta = lambda_measure.generate_points(1)
            numberOfSeatedCostumers.append(1.)
            Thetas.append(theta)
            
        else:
            numberOfSeatedCostumers[selectedTable] = numberOfSeatedCostumers[selectedTable] + 1.
    
    return (Thetas,numberOfSeatedCostumers)