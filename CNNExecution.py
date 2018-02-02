import CNNmnist as cnn
import math
from tensorflow.examples.tutorials.mnist import input_data
import time
import copy


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
    take_best_nr = 10 #Rodo kiek individų imsiu į train Full Network etapą.
    batch_size_test = 20 #Į kiek dalių skaidyti testavimo dataset (ne mažinti, gali išmesti errorą dėl atminties trūkumo)
    training_time_limit = 24000 #Treniravimo laiko limitas. Jei bus viršyta, treniravimas bus nutrauktas ir accuracy nustatomas į 0
    data_set = input_data.read_data_sets('MNIST_data', one_hot=True)

    def sortNetworksByFitness(self, past_generation_networks, fitness):
        # Istatomas fitnessas ir pagal ji rikiuojama
        for past_gen_net_idx in range(0, len(past_generation_networks)):
            past_generation_networks[past_gen_net_idx].insert(0, fitness[past_gen_net_idx])
        past_generation_networks.sort(key=lambda x: x[0], reverse=True)

        return past_generation_networks

    def __init__(self, example_count_train=20000):
        try:
            self.example_count_train = example_count_train  # Su kiek nuotraukų treniruosime individą
        except Exception as error:
            print('Caught this error: ' + repr(error))

    #Metodas naudojamas, kai treniruojame tik paskutinį layer,
    #kad padidintų šansą keliauti į Full Network etapą tuos network, kurie yra didesni.
    #Kadangi mažo ilgio networkai (arba networkai su daug pool layeriu) rodo gerus rezultatus treniruojant tik paskutinį layerį.
    def get_additional_score(self, individual, result):
        number_of_pool = 0
        for layer in individual:
            if layer[0] == "APool" or layer[0] == "MPool":
                number_of_pool += 1

        if len(individual) < 4:
            result -= 20
        if len(individual) >= 6 & number_of_pool < 3:
            result += 10
        if result < 0:
            result = 0

        return result

    #Gražina masyvą su individų iš populiacijos tikslumu. Rezultato masyvo indeksai atitinka paduotų individų indeksus populiacijoje.
    def evaluate_population(self, population, start_only_last_layer = True, nr=-1):
        print(population)
        if nr != -1:
            print("Generation nr. ", nr)
        rez = []
        batch_size_train = 50
        t0 = time.time()
        a = 1
        if start_only_last_layer:
            for individual in population:
                print("-----------------------------------------------------------------------------------")
                print("Training individual nr.", a, "(Only last layer)")
                print(str(individual))
                iteration_count = math.floor(self.example_count_train / batch_size_train)
                print("Number of steps " + str(iteration_count))
                result = cnn.CNN(individual, iteration_count, batch_size_train, self.batch_size_test, self.training_time_limit, self.data_set, True).exec_cnn()
                if result == 0:
                    result = cnn.CNN(individual, iteration_count, batch_size_train, self.batch_size_test,
                                     self.training_time_limit, self.data_set, True).exec_cnn()

                result = self.get_additional_score(individual, result)

                rez.append(result)
                a += 1

            print("-----------------------------------------------------------------------------------")
            for i in range(len(population)):
                print(population[i])
                print(rez[i])
            print("-----------------------------------------------------------------------------------")
            op = copy.deepcopy(self.sortNetworksByFitness(copy.deepcopy(population), rez))

            to_next_phase = self.take_best_nr
            if len(population) < self.take_best_nr:
                to_next_phase = len(population)

            print("-----------------------------------------------------------------------------------")
            print("TOP " + str(to_next_phase) + " going to the next round:")

            for i in range(to_next_phase):
                print(op[i])
                print("-----------------------------------------------------------------------------------")
            a = 1
        else:
            to_next_phase = population

        for i in range(to_next_phase):
            print("Training individual nr.", a, "(All layers)")

            op[i].pop(0)
            print(op[i])
            ind = population.index(op[i])
            rez.pop(ind)
            ab = population.pop(ind)
            population.append(ab)
            result = cnn.CNN(op[i], iteration_count, batch_size_train, self.batch_size_test, self.training_time_limit,
                        self.data_set, False).exec_cnn()
            if result == 0:
                result = cnn.CNN(op[i], iteration_count, batch_size_train, self.batch_size_test, self.training_time_limit,
                                 self.data_set, False).exec_cnn()
            rez.append(result)
            a += 1

        print("Laikas - ", time.time() - t0)

        print(rez)

        return rez

