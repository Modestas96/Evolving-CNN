#Čia atrodo +/- viskas veikia. Vėliau parašysiu komentarus prie kodo dalių

import random
import execProgram
from deap import base
from deap import creator
from deap import tools
import numpy as np
import math

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 74)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def writeRezToFile(individual, fitness, g):
    f = open("test.txt", "a+")  # opens file with name of "test.txt"
    f.write("Generation: %g \n" % g)
    f.write("MaxFitness: %g \n" % fitness)
    f.write(individual)
    f.write("\n")
    f.write(indiFormat(individual))
    f.close()


def indiFormat(individual):
    FCLayer = individual[65:74]
    FCNodeCount = int("".join(str(x) for x in FCLayer), 2)
    Layers = np.reshape(individual[0:65], (5, 13))

    CurrentDim = 28

    LayersOut = []
    prevType = -1
    convCountRow = 0
    for i in range(len(Layers)):
        Lay = Layers[i]
        LayerType = int("".join(str(x) for x in Lay[0:2]), 2)
        if LayerType == 0 | LayerType == 1:
            LayerType = -1
        else:
            LayerType -= 2
        KernelSizeBin = Lay[2:5]
        if i == 0:
            LayerType = 0
        if convCountRow >= 2:
            LayerType = 1
        if LayerType == 1:
            KernelSizeBin = Lay[2:3]

        if LayerType == -1:
            continue

        KernelSize = int("".join(str(x) for x in KernelSizeBin), 2)

        if prevType == 1 & LayerType == 1:
            LayerType = 0

        prevType = LayerType

        if LayerType == 1:
            KernelSize += 2
            nextDim = math.ceil(CurrentDim / (KernelSize))
            if nextDim < 5:
                LayerType = 0
            else:
                CurrentDim = nextDim
                LayersOut.append(["Pool", (KernelSize)])
                convCountRow = 0
        if LayerType == 0:
            depthBin = Lay[5:11]
            depth = int("".join(str(x) for x in depthBin), 2)
            if depth < 8:
                depth = 8
            if KernelSize < 2:
                KernelSize = 2
            LayersOut.append(["Conv", KernelSize, depth])
            convCountRow += 1

    if FCNodeCount < 25:
        FCNodeCount = 25
    LayersOut.append(["FC", int(math.pow(math.ceil(math.sqrt(FCNodeCount)), 2)), 1])

    return LayersOut


def evalOneMax(individual):
    LayersOut = indiFormat(individual)
    fitnesss = execProgram.runCNN(LayersOut)
    return fitnesss,


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=10)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while max(fits) < 100 and g < 20:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.5:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


if __name__ == "__main__": main()

