import random
from copy import deepcopy

import CNNExecution as CNN
from time import gmtime, strftime

#Sukuriu failų pavadinmus.
time = strftime("%Y%m%d%H%M%S", gmtime())
fileName1 = "Results"+time+".txt"
fileName2 = "LastPopulations"+time+".txt"


#Spausdina į failą nurodytą individą ir jo fitness. g - generacijos nr.
def write_res_to_file(individual, fitness, g):
    f = open(fileName1, "a+")  # opens file with name of "test.txt"
    f.write("Generation: %g \n" % g)
    f.write("MaxFitness: %g \n" % fitness)
    f.write("\n")
    f.write(str(individual))
    f.write("\n")
    f.write("--------------------------------------------------")
    f.write("\n")
    f.close()


#Spausdina į failą nurodytą populiaciją, tik paskutinę
# Nu jei gausime errorą, bent jau žinosim kurioje populiacijoje gavome jį.
# (Darau prielaida, kad po erroro programa sustos).
def write_debug_to_file(population, g):
    f = open(fileName2, "w+")  # opens file with name of "test.txt"
    f.write("Generation: " + str(g) + "\n")
    for ind in population:
        f.write(str(ind) + "\n")

    f.write("--------------------------------------------------")
    f.write("\n")
    f.close()

# Metodas generuoja naujus sluoksnius atsitiktinai
# Grazina visus sukurtus tinklus
def startGenerateFromScratch(population, input_length, max_layer_amount, max_conv_depth, max_fc_size,
                             max_pool_kernel_size, max_conv_kernel_size):
    # Random sugeneruoti netai
    NETWORKS = []

    for network in range(0, population):
        # Sluoksniu skaicius niekada nemazesnis uz 2, nes pirmas yra Conv sluoksnis, paskutinis FC
        layer_amount = random.randint(2, max_layer_amount)
        # Laikomas dabartines iteracijos tinklas
        net = []

        # Issaugojami dydziai, kad paduoti metodo parametrai nebutu keiciami
        cur_input_length = input_length
        # Jei tinkle buvo FC sluoksnis, po jo daryti tik taip pat FC sluoksnius
        fc_passed = False

        # Kurti sluoksnius dabartines iteracijos netui
        for layerNum in range(0, layer_amount):
            # Laikomas dabartinis sluoksnis
            layer = []

            # Jei pirmas sluoksnis, jis butinai turi buti Conv
            if layerNum == 0:
                layer = startGenerateConvLayer(input_length, max_conv_depth, max_conv_kernel_size)
            # Jei paskutinis sluoksnis, jis butinai turi buti FC
            elif layerNum == layer_amount - 1:
                layer = startGenerateFCLayer(max_fc_size)
            # Kitu atveju atsitiktinai generuoti sluoksni ir jo tipa
            else:
                layer_type = random.randint(0, 2)

                # Jei sluoksnio tipas 0, kurti Conv tipo sluoksni
                if layer_type == 0 and not fc_passed:
                    layer = startGenerateConvLayer(cur_input_length, max_conv_depth, max_conv_kernel_size)
                    # po conv layerio x ir y lieka nepakite
                    cur_input_length = cur_input_length

                # Jei sluoksnio tipas 1, kurti (max) Pool tipo sluoksni
                elif layer_type == 1 and not fc_passed:
                    layer = startGeneratePoolLayer(cur_input_length, max_pool_kernel_size)
                    # stride toks pats kaip kSize, todel x ir y bus dalinamas pagal Pool stride
                    cur_input_length = cur_input_length / layer[1]
                # Jei sluoksnio tipas 2, kurti FC tipo sluoksni
                elif layer_type == 2 or fc_passed:
                    layer = startGenerateFCLayer(max_fc_size)
                    fc_passed = True

            net.append(layer)
        net = fix_network(net, max_layer_amount, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
                          max_conv_kernel_size, False)
        NETWORKS.append(net)
    return NETWORKS


# Metodas atsitiktinai kuria Conv sluoksni
# Grazina sluoksnio masyva
def startGenerateConvLayer(input_length, max_depth, max_conv_kernel_size):
    LAYER = ["Conv", random.randint(1, int(min(input_length, max_conv_kernel_size))), random.randint(1, max_depth)]
    return LAYER


# Metodas atsitiktinai kuria Pool (APool arba MPool su 50 proc. sansu atskirai) sluoksni
# Grazina sluoksnio masyva
def startGeneratePoolLayer(input_length, max_pool_kernel_size):
    type = random.randint(0, 1)
    if type == 0:
        LAYER = ["APool"]
    else:
        LAYER = ["MPool"]
    kSize = random.randint(1, int(min(input_length, max_pool_kernel_size)))
    LAYER.append(kSize)
    return LAYER


# Metodas atsitiktinai kuria Average Pool sluoksni
# Grazina sluoksnio masyva
def startGenerateAvgPoolLayer(input_length, max_pool_kernel_size):
    LAYER = ["APool"]
    kSize = random.randint(1, int(min(input_length, max_pool_kernel_size)))
    LAYER.append(kSize)
    return LAYER


# Metodas atsitiktinai kuria Max Pool sluoksni
# Grazina sluoksnio masyva
def startGenerateMaxPoolLayer(input_length, max_pool_kernel_size):
    LAYER = ["MPool"]
    kSize = random.randint(1, int(min(input_length, max_pool_kernel_size)))
    LAYER.append(kSize)
    return LAYER


# Metodas atsitiktinai kuria FC sluoksni
# Grazina sluoksnio masyva
def startGenerateFCLayer(max_fc_size):
    LAYER = ["FC", random.randint(10, max_fc_size), 1]
    return LAYER

# Surikiuoti tinklus mazejimo tvarka pagal ju geruma (fitnessa)
def sortNetworksByFitness(past_generation_networks, fitness):
    # Istatomas fitnessas ir pagal ji rikiuojama
    for past_gen_net_idx in range(0, len(past_generation_networks)):
        past_generation_networks[past_gen_net_idx].insert(0, fitness[past_gen_net_idx])
    past_generation_networks.sort(key=lambda x: x[0], reverse=True)

    # Fitness isimamas is masyvo
    for past_gen_net_idx in range(0, len(past_generation_networks)):
        past_generation_networks[past_gen_net_idx].pop(0)

    return past_generation_networks


# Metodas grazina geriausius netus ir praeito generationo
def returnBest(past_generation_networks, fitness, best_unchanged_amount):
    # Geriausiu atrinktu masyvas
    best_networks = []

    # Pridedamas i kito generationo masyva
    for best_unchanged_idx in range(0, best_unchanged_amount):
        best_networks.append(deepcopy(past_generation_networks[best_unchanged_idx]))

    return best_networks



# Metodas parenka sekancios kartos netus, renka, kurie bus kryzminami
# Grazina kitos kartos tinklu masyva
def generationSelection(past_generation_networks, crossover_amount, random_new_amount,
                        max_layer_amount, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
                        max_conv_kernel_size):

    # Kito generationo masyvas
    next_generation_networks = []

    # Sukurti random_new_amount nauju atsitiktinu netu ir ideti juos i kita generationa
    random_new_networks = startGenerateFromScratch(random_new_amount, input_length, max_layer_amount, max_conv_depth,
                                                   max_fc_size, max_pool_kernel_size, max_conv_kernel_size)
    for random_new_network in random_new_networks:
        next_generation_networks.append(random_new_network)


    # Rinkti kombinacijas tinklu ju kryzminimui, tikrinti
    # Galimos ir tos pacios kombinacijos, kadangi kryzminimas yra su atsitiktinumais, nesigaus identiskas rezultatas
    for crossover_idx in range(0, crossover_amount):
        parent1_idx = min(int(abs(random.gauss(0, 5))), len(past_generation_networks) - 1)
        parent2_idx = min(int(abs(random.gauss(0, 5))), len(past_generation_networks) - 1)
        cross_net = crossover1(deepcopy(past_generation_networks[parent1_idx]),
                               deepcopy(past_generation_networks[parent2_idx]),
                               max_layer_amount, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
                               max_conv_kernel_size)
        mutated_cross_net = mutation(cross_net, 0.05, 0.1, 0.2, max_layer_amount, max_conv_depth, max_fc_size,
                                     input_length, max_pool_kernel_size, max_conv_kernel_size)
        mutated_cross_net = fix_network(mutated_cross_net, max_layer_amount, max_conv_depth, max_fc_size, input_length,
                                        max_pool_kernel_size, max_conv_kernel_size, True)
        next_generation_networks.append(mutated_cross_net)


    return next_generation_networks


# Metodas apskaiciuoja (iskviecia skaicavimo metoda) kiekvieno tinklo fitness function
# Grazina masyva su fitnessu kiekvienam tinklui atitinkamose masyvo pozicijose
def calculateFitness(networks):
    # !! TUSCIAS METODAS !!
    # Sukurti masyva, tokio dydzio kaip networks, jame ikelti fitnessai, pozicijos masyve tos pacios atitinkamam
    # tinklui networks masyve
    fitness = CNN.runCNN(networks)
    return fitness


# Metodas is dvieju pasirinktu tinklu padaro sukryzminta
def crossover1(parent1, parent2, max_layer_amount, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
               max_conv_kernel_size):
    # Crossover1 veikimas- kazkokia dalis parent1 tinklo yra istrinama, i ta vieta yra perkialiama kazkokia
    # parent2 tinklo dalis. Pirmas Conv layeris ir paskutinis FC layeris yra irgi pakeiciami arba suvidurkinami

    # cross_net bus talpinamas naujas netas
    cross_net = parent1[:]

    # Kuriami random range, is kurio is parent 2 bus paimami sluoksniai ir perkeliami i parent 1 sluoksniu vietas
    while True:
        rand_range_1 = []
        rand_range_2 = []

        # Kuriamas range, is kurio bus isimami sluoksniai parent1
        for rand_num in range(0, 2):
            if len(parent1) > 2:
                rand_range_1.append(random.randint(1, len(parent1) - 2))
            else:
                rand_range_1 = [1, 1]
        # Kuriamas range, is kurio bus perkeliami sluoksniai is parent2 i parent1
        for rand_num in range(0, 2):
            if len(parent2) > 2:
                rand_range_2.append(random.randint(0, len(parent2) - 1))
            else:
                rand_range_2 = [1, 1]

        # Jei bus virsytas sluoksniu limitas tai generuoti skaicius is naujo
        if len(cross_net) - (max(rand_range_1) - min(rand_range_1)) + (max(rand_range_2) - min(rand_range_2)) <= \
                max_layer_amount:
            break

    # Trinami sluoksniai parent1
    for layer_idx in range(0, max(rand_range_1) - min(rand_range_1)):
        cross_net.pop(min(rand_range_1))

    # Ikeliami sluoksniai is parent2 i parent1
    for layer_idx in range(min(rand_range_2), max(rand_range_2)):
        cross_net.insert(min(rand_range_1) + (layer_idx - min(rand_range_2)), parent2[layer_idx])

    # Sukryzmint pirma ir paskutini (conv ir fc) layerius, apkeiciant arba sujungiant mazdaug vidurkiais
    crossover_first_layer_type = random.randint(0, 1)
    if crossover_first_layer_type == 0:
        cross_net = crossover_first_last_layers_swap(cross_net, parent2)
    else:
        cross_net = crossover_first_last_layers_merge(cross_net, parent1, parent2, max_conv_depth, max_fc_size,
                                                      input_length, max_conv_kernel_size)

    # Sutvarkyt neta, kad nebutu netinkamu dydziu struktura, nebutu po fc layerio conv arba pool
    # cross_net = fix_network(cross_net, max_layer_amount, max_conv_depth, max_fc_size, input_length,
    #                        max_pool_kernel_size, max_conv_kernel_size, True)

    return cross_net


# Metodas sutvarko neuroninio tinklo struktura. Jei po mutaciju ar kryzminimo neatitinka layerio seka, pvz pooling
# kernelio dydis dalinasi su liekana is inputo dimensiju, arba yra kitokiu layeriu po fc layerio, pertvarkyti
# taip, kad tas nebutu pazeista
def fix_network(network, max_layer_size, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
                max_conv_kernel_size, change_fc):
    # Issaugojamas input_length, kuris bus keiciamas sluoksniu tikrinimo eigoje
    cur_input_length = input_length
    # Tikrinimui, ar fc layeris buvo esant kitiems layeriams po jo
    fc_passed = False

    # PAPILDOMI TIKRINIMAI (ju paciu nereikia, jei nera bugu, bent kol kas):
    # 1. Jei networkas turi daugiau layeriu nei limitas- trinti viena pries limita ir visus kitus po, pergeneruot
    #    random paskutini FC
    # 2. Jei pirmas layeris ne conv- generuoti random pirma conv
    # 3. Jei paskutinis layeris ne FC- generuoti random paskutini FC

    for layer_idx in range(1, len(network) - 1):
        # Jei conv tipo layeris
        if network[layer_idx][0] == "Conv":
            # Jei pries ji buvo FC layeris tai layeri perrasyti random fc layeriu
            if fc_passed:
                network[layer_idx] = startGenerateFCLayer(max_fc_size)
                continue

            # Jei kernelio dydis per mazas arba per didelis
            if network[layer_idx][1] < 1:
                network[layer_idx][1] = 1
            elif network[layer_idx][1] > min(input_length, max_conv_kernel_size):
                network[layer_idx] = min(input_length, max_conv_kernel_size)

            # Jei kerneliu depth per mazas arba per didelis
            if network[layer_idx][2] < 1:
                network[layer_idx][2] = 1
            elif network[layer_idx][2] > max_conv_depth:
                network[layer_idx] = max_conv_depth

        # Jei pool tipo layeris
        elif network[layer_idx][0] == "APool" or network[layer_idx][0] == "MPool":
            # Jei pries ji buvo FC layeris tai layeri perrasyti random fc layeriu
            if fc_passed:
                network[layer_idx] = startGenerateFCLayer(max_fc_size)
                continue

            # Jei kernelio dydis per mazas arba per didelis
            print(network)
            if network[layer_idx][1] < 1:
                network[layer_idx][1] = 1
            elif network[layer_idx][1] > min(input_length, max_pool_kernel_size):
                network[layer_idx] = min(input_length, max_pool_kernel_size)

            # Pakeisti inputo ilgi sekanciam layeriui
            cur_input_length = cur_input_length / network[layer_idx][1] + 1

        elif network[layer_idx][0] == "FC":
            fix_fc = random.randint(0, 1)
            if fix_fc == 0 and not fc_passed and change_fc:
                # layer_type = random.randint(0, 1)
                # if layer_type == 0:
                #     network[layer_idx] = startGenerateConvLayer(input_length, max_conv_depth,
                #                                                 int(min(max_conv_kernel_size, input_length)))
                # else:
                #    network[layer_idx] = startGeneratePoolLayer(input_length,
                #                                                int(min(max_pool_kernel_size, input_length)))
                network[layer_idx] = network[layer_idx + 1]
                if network[layer_idx][0] == "FC":
                    fc_passed = True
                layer_idx -= 1
            else:
                # Jei fc nodu kiekis mazesnis uz 1, padaryti 1
                if network[layer_idx][1] < 10:
                    network[layer_idx][1] = 10

                # Jei fc nodu kiekis didesnis uz limita, padaryti ju skaiciu limito skaiciumi
                elif network[layer_idx][1] > max_fc_size:
                    network[layer_idx][1] = max_fc_size

                # Uztvirtinama, kad einant per sluoksnius FC sluoksnis jau buvo
                fc_passed = True

    # Trinti FC layerius, jei ju pries pabaiga daugiau nei 2
    fc_amount = 0
    idx_layer = len(network) - 1
    while network[idx_layer][0] == "FC":
        fc_amount += 1
        idx_layer -= 1

    fc_amount -= 2

    while fc_amount > 0:
        network.pop(len(network) - 1)
        fc_amount -= 1

    return network


# Parenkamas "vidurkis" pagal gauso skirstini tarp abieju layeriu parent1 ir parent2
def crossover_first_last_layers_merge(cross_net, parent1, parent2, max_conv_depth, max_fc_size, input_length, max_conv_kernel_size):
    conv_kSize = 0
    # Jei kernel size didesnis uz inputo dydi ar mazesnis uz 1, generuoti skaiciu is naujo
    while conv_kSize > min(input_length, max_conv_kernel_size) or conv_kSize < 1:
        rand = int(random.gauss(0, (max(parent1[0][1], parent2[0][1]) - min(parent1[0][1], parent2[0][1])) / 2))
        conv_kSize = int(min(parent1[0][1], parent2[0][1]) + rand)

    conv_depth = 0
    # Jei kernel depth yra daugiau uz limita ar maziau uz 1, generuoti skaiciu is naujo
    while conv_depth > max_conv_depth or conv_depth < 1:
        rand = int(random.gauss(0, (max(parent1[0][2], parent2[0][2]) - min(parent1[0][2], parent2[0][2])) / 2))
        conv_depth = int(min(parent1[0][2], parent2[0][2]) + rand)

    fc_size = 0
    # Jei fc nodu kiekis mazesnis uz 1 arba didenis uz limita, generuoti skaiciu is naujo
    while fc_size > max_fc_size or fc_size < 1:
        rand = int(random.gauss(0, (max(parent1[len(parent1) - 1][1], parent2[len(parent2) - 1][1])
                                - min(parent1[len(parent1) - 1][1], parent2[len(parent2) - 1][1])) / 2))
        fc_size = min(parent1[len(parent1) - 1][1], parent2[len(parent2) - 1][1]) + rand

    cross_net[0][1] = conv_kSize
    cross_net[0][2] = conv_depth
    cross_net[len(cross_net) - 1][1] = fc_size
    return cross_net


# Metodas sukryzmina pirma ir paskutini sluoksni pakeisdamas viena kitu is parent1 i parent2
def crossover_first_last_layers_swap(cross_net, parent2):
    # 50/50, kad pirmas Conv layeris bus pakeistas is parent1 layerio i parent2
    if random.randint(0, 1) == 0:
        cross_net.pop(0)
        cross_net.insert(0, parent2[0])
    # 50/50 kad paskutinis FC layeris bus pakeistas is parent1 layerio i parent2
    if random.randint(0, 1) == 0:
        cross_net.pop(len(cross_net) - 1)
        cross_net.append(parent2[len(parent2) - 1])
    return cross_net


# Metodas mutuoja neuronini tinkla
def mutation(network, layer_type_change_prob, mid_layer_data_change_prob, end_start_layer_data_change_prob,
             max_layer_size, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size, max_conv_kernel_size):
    # Apskaiciuoti random, ar bus atliekama mutacija
    rand_type_change = random.randint(0, 100) / 100
    rand_mid_data_change = random.randint(0, 100) / 100
    rand_end_start_data_change = random.randint(0, 100) / 100

    # Sluoksnio tipo keitimo mutacija bus vykdoma, jei netas turi bent viena vidini sluoksni tarp pirmo conv ir pask. fc
    if rand_type_change <= layer_type_change_prob and len(network) - 2 > 0:
        layer = random.randint(0, len(network) - 3)
        type = random.randint(0, 2)
        if type == 0:
            network[layer + 1] = startGenerateConvLayer(input_length, max_conv_depth, max_conv_kernel_size)
        elif type == 1:
            network[layer + 1] = startGeneratePoolLayer(input_length, max_pool_kernel_size)
        else:
            network[layer + 1] = startGenerateFCLayer(max_fc_size)

    # Keičiami tik sluoksnio duomenys, tipas paliekamas tas pats
    if rand_mid_data_change <= mid_layer_data_change_prob:
        layer = random.randint(1, len(network) - 2)
        if network[layer][0] == "Conv":
            rand = int(random.gauss(0, max_conv_kernel_size / 8))
            while network[layer][1] + rand > max_conv_kernel_size or network[layer][1] + rand < 1:
                rand = int(random.gauss(0, max_conv_kernel_size / 8))
            network[layer][1] += rand
            rand = int(random.gauss(0, max_conv_depth / 8))
            while network[layer][2] + rand > max_conv_depth or network[layer][2] + rand < 1:
                rand = int(random.gauss(0, max_conv_kernel_size / 8))
            network[layer][2] += rand
        elif network[layer][0] == "APool":
            network[layer][1] = startGenerateAvgPoolLayer(input_length, max_pool_kernel_size)
        elif network[layer][0] == "MPool":
            network[layer] = startGenerateMaxPoolLayer(input_length, max_pool_kernel_size)
        else:
            rand = int(random.gauss(0, max_fc_size / 8))
            while network[layer][1] + rand > max_fc_size or network[layer][1] + rand < 10:
                rand = int(random.gauss(0, max_fc_size / 8))
            network[layer][1] += rand

    # Keiciami tik pirmo arba paskutnio sluoksnio duomenys, tipas paliekamas tas pats (del tikimybiu skirtumo)
    if rand_end_start_data_change <= end_start_layer_data_change_prob:
        layer = random.randint(0, 1)
        if layer == 0:
            rand = int(random.gauss(0, max_conv_kernel_size / 8))
            while network[layer][1] + rand > max_conv_kernel_size or network[layer][1] + rand < 1:
                rand = int(random.gauss(0, max_conv_kernel_size / 8))
            network[layer][1] += rand
            rand = int(random.gauss(0, max_conv_depth / 8))
            while network[layer][2] + rand > max_conv_depth or network[layer][2] + rand < 1:
                rand = int(random.gauss(0, max_conv_kernel_size / 8))
            network[layer][2] += rand
        else:
            rand = int(random.gauss(0, max_fc_size / 8))
            while network[layer][1] + rand > max_fc_size or network[layer][1] + rand < 10:
                rand = int(random.gauss(0, max_fc_size / 8))
            network[layer][1] += rand

    # network = fix_network(network, max_layer_size, max_conv_depth, max_fc_size, input_length, max_pool_kernel_size,
    #                       max_conv_kernel_size, False)
    return network


# Metodas random priskiria skaiciu fitnessui
def test_random_fitness(array_size):
    fitness = []
    for i in range(0, array_size):
        fitness.append(random.randint(0, 1000000) / 10000)
    return fitness


def main():
    g = 1

    elitism_amount = 5
    random_new_amount = 3
    crossover_amount = 22
    starting_amount = 30

    input_length = 28
    layer_limit = 8
    conv_depth_limit = 64
    fc_node_limit = 999
    conv_kernel_limit = 13
    pool_kernel_limit = 4

    BEST_NETWORKS = []
    BEST_FITNESS = []

    do_print = True  # Ar spausdinti į failus?
    NETWORKS = startGenerateFromScratch(starting_amount, input_length, layer_limit, conv_depth_limit, fc_node_limit,
                                        pool_kernel_limit, conv_kernel_limit)
    if do_print:
        write_debug_to_file(NETWORKS, g)

    # Treniruoti ir testuoti tinklus
    #FITNESS = CNN.CNNExecution().evaluate_population(NETWORKS, False)
    FITNESS = test_random_fitness(30)
    while True:

        max_fit_ind = FITNESS.index(max(FITNESS))
        if do_print:
            write_res_to_file(NETWORKS[max_fit_ind], FITNESS[max_fit_ind], g)
        g += 1

        # Tinklai rikiuojami mazejimo tvarka pagal fitnessa
        NETWORKS = sortNetworksByFitness(NETWORKS, FITNESS)
        FITNESS.sort(reverse=True)

        # Grazina geriausius networkus is praeito generationo
        BEST_NETWORKS = returnBest(NETWORKS, FITNESS, elitism_amount)
        # Geriausiu networku fitnessas issaugojamas, ju traininti dar karta nereikia
        for idx in range(0, elitism_amount):
            BEST_FITNESS.append(FITNESS[idx])

        # Generuoti kita karta
        NETWORKS = generationSelection(NETWORKS, crossover_amount, random_new_amount, layer_limit, conv_depth_limit,
                                       fc_node_limit, input_length, pool_kernel_limit, conv_kernel_limit)

        for idx in range(0, len(NETWORKS)):
            print(NETWORKS[idx])

        # Treniruoti ir testuoti tinklus
        FITNESS = CNN.CNNExecution().evaluate_population(NETWORKS, False)


        # Ideti praeitos kartos geriausius kartu su fitnessu
        for idx in range(0, elitism_amount):
            FITNESS.append(BEST_FITNESS[idx])
            NETWORKS.append(deepcopy(BEST_NETWORKS[idx]))

        for idx in range(0, len(NETWORKS)):
            print(NETWORKS[idx])
            print(FITNESS[idx])

        BEST_NETWORKS = []
        BEST_FITNESS = []

        if do_print:
            write_debug_to_file(NETWORKS, g)
'''
def main():
    g = 1
    do_print = False  #Ar spausdinti į failus?
    NETWORKS = startGenerateFromScratch(30, 28, 6, 64, 1000, 4, 13)
    if do_print:
        write_debug_to_file(NETWORKS, g)
    training_examples = 200 # Su kiek training example iš pradžių skaičiuti?
    inc_tr_ex_times = 2  # Kiek kartų padidinti esamą training example skaičių?
    inc_tr_ex_every_gen = 20  # Kas kiek generacijų padidinti training example skaičių?
    while True:
        if g % inc_tr_ex_every_gen == 0:
            training_examples *= inc_tr_ex_times
        for idx in range(0, len(NETWORKS)):
            print(NETWORKS[idx])
        fitness = CNN.CNNExecution(training_examples).evaluate_population(NETWORKS)
        for idx in range(0, len(NETWORKS)):
            print(NETWORKS[idx])
            print(fitness[idx])
        max_fit_ind = fitness.index(max(fitness))
        if do_print:
            write_res_to_file(NETWORKS[max_fit_ind], fitness[max_fit_ind], g)
        g += 1
        NETWORKS = generationSelection(NETWORKS, fitness, 5, 17, 3, 6, 64, 1000, 28, 4, 13)
        if do_print:
            write_debug_to_file(NETWORKS, g)
'''

if __name__ == "__main__":
    main()
