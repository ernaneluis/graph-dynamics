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

def ExtendedChineseRestaurantProcess(numberOfCostumers,lambda_measure,tables_and_costumers):
    """
    Chinese restaurant process
    """
    c_star = sum(tables_and_costumers.values())
    alpha = lambda_measure.interval_size
    costumer_seats = []
    Thetas = []
    p = []
    numberOfSeatedCostumers = []
    for table, number_of_costumers in tables_and_costumers.iteritems():
        Thetas.append(table)
        p.append(number_of_costumers/(alpha + c_star))
        numberOfSeatedCostumers.append(number_of_costumers)
    p.append(alpha/(alpha + c_star))
    numberOfTables = len(tables_and_costumers.keys())
        
    for i in range(numberOfCostumers):
        p = np.concatenate([np.array(numberOfSeatedCostumers),[alpha]])/(alpha + c_star + sum(numberOfSeatedCostumers))
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