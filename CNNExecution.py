import CNNmnist as cnn
import math
from tensorflow.examples.tutorials.mnist import input_data
'''
Kaip naudoti:
     1. Sukuri objektą CNNExecution.CNNExecution()
     2. Paleidi metodą evaluate_population(population): population - Trimatis visos populiacijos masyvas
     3. evaluate_population gražinamas rezultas yra masyvas su individo accuracy po treniravimo
     
Pavyzdys:
    import CNNExecution as CNNexe
    
    fitness = []
    
    NETWORKS = ...
    
    fitness = CNNexe.CNNExecution().evaluate_population(Networks)
    
    print(fitness): [89.1235, 90.321, 92.1253, 70.1234, 95.3134]       
'''

class CNNExecution:
    batch_size_test = 20 #Į kiek dalių skaidyti testavimo dataset (ne mažinti, gali išmesti errorą dėl atminties trūkumo)
    training_time_limit = 240 #Treniravimo laiko limitas. Jei bus viršyta, treniravimas bus nutrauktas ir accuracy nustatomas į 0
    data_set = input_data.read_data_sets('MNIST_data', one_hot=True)

    def __init__(self, example_count_train=20000):
        try:
            self.example_count_train = example_count_train  # Su kiek nuotraukų treniruosime individą
        except Exception as error:
            print('Caught this error: ' + repr(error))

    #Gražina masyvą su individų iš populiacijos tikslumu. Rezultato masyvo indeksai atitinka paduotų individų indeksus populiacijoje.
    def evaluate_population(self, population, nr=-1):
        if nr != -1:
            print("Generation nr. ", nr)
        rez = []
        i = 1
        for individual in population:
            print("-----------------------------------------------------------------------------------")
            #print("Individual nr. ", i)
            print(str(individual))
            batch_size_train = 64
            if batch_size_train <= 0:
                print("Batch size must be greater than 0, skipping this")
                continue
            iteration_count = math.floor(self.example_count_train / batch_size_train)
            print("Number of steps " + str(iteration_count))
            rez.append(cnn.CNN(individual, iteration_count, batch_size_train, self.batch_size_test, self.training_time_limit, self.data_set).exec_cnn())
            i += 1
        return rez

