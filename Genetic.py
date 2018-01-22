import random
import execProgram as CNN

# Metodas generuoja naujus sluoksnius atsitiktinai
# Grazina visus sukurtus tinklus
def startGenerateFromScratch(population, input_length, max_layer_amount, max_conv_depth, max_fc_size):
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
                layer = startGenerateConvLayer(input_length, max_conv_depth)
            # Jei paskutinis sluoksnis, jis butinai turi buti FC
            elif layerNum == layer_amount - 1:
                layer = startGenerateFCLayer(max_fc_size)
            # Kitu atveju atsitiktinai generuoti sluoksni ir jo tipa
            else:
                layer_type = random.randint(0, 2)

                # Jei sluoksnio tipas 0, kurti Conv tipo sluoksni
                if layer_type == 0 and not fc_passed:
                    layer = startGenerateConvLayer(cur_input_length, max_conv_depth)
                    # po conv layerio x ir y lieka nepakite
                    cur_input_length = cur_input_length

                # Jei sluoksnio tipas 1, kurti (max) Pool tipo sluoksni
                elif layer_type == 1 and not fc_passed:
                    layer = startGeneratePoolLayer(cur_input_length)
                    # stride toks pats kaip kSize, todel x ir y bus dalinamas pagal Pool stride
                    cur_input_length = cur_input_length / layer[1]
                # Jei sluoksnio tipas 2, kurti FC tipo sluoksni
                elif layer_type == 2 or fc_passed:
                    layer = startGenerateFCLayer(max_fc_size)
                    fc_passed = True

            net.append(layer)
        NETWORKS.append(net)
    return NETWORKS


# Metodas atsitiktinai kuria Conv sluoksni
# Grazina sluoksnio masyva
def startGenerateConvLayer(input_length, max_depth):
    LAYER = ["Conv", random.randint(1, input_length), random.randint(1, max_depth)]
    return LAYER


# Metodas atsitiktinai kuria Pool sluoksni
# Grazina sluoksnio masyva
def startGeneratePoolLayer(input_length):
    LAYER = ["Pool"]
    kSize = input_length + 1
    while input_length % kSize != 0:
        kSize = random.randint(1, input_length)
    LAYER.append(kSize)
    return LAYER


# Metodas atsitiktinai kuria FC sluoksni
# Grazina sluoksnio masyva
def startGenerateFCLayer(max_fc_size):
    LAYER = ["FC", random.randint(0, max_fc_size), 1]
    return LAYER


# Metodas parenka sekancios kartos netus, renka, kurie bus kryzminami
# Grazina kitos kartos tinklu masyva
def generationSelection(past_generation_networks, fitness, best_unchanged_amount, crossover_amount, random_new_amount):
    # Kito generationo masyvas
    next_generation_networks = []

    # Sukurti random_new_amount nauju atsitiktinu netu ir ideti juos i kita generationa
    random_new_networks = startGenerateFromScratch(random_new_amount, 28, 6, 64, 1000)
    for random_new_network in random_new_networks:
        next_generation_networks.append(random_new_network)

    # Rinkti kombinacijas tinklu ju kryzminimui, tikrinti
    # Galimos ir tos pacios kombinacijos, kadangi kryzminimas yra su atsitiktinumais, nesigaus identiskas rezultatas
    parent1_idx = abs(random.gauss(0, 5))
    parent2_idx = abs(random.gauss(0, 5))
    for crossover_idx in range(0, crossover_amount):
        next_generation_networks.append(
            crossover1(past_generation_networks[parent1_idx], past_generation_networks[parent2_idx]))

    # Istatyti i kita generationa best_unchanged_amount kieki geriausiu netu
    # Fitness pridedamas kaip pirmas masyvo elementas, pagal ji rikiuojamas masyvas
    for past_gen_net_idx in range(0, len(past_generation_networks)):
        past_generation_networks[past_gen_net_idx].insert(0, fitness[past_gen_net_idx])
    past_generation_networks.sort(key=lambda x: x[0], reverse=True)
    # Fitness isimamas is masyvo ir pridedamas i kito generationo masyva
    for best_unchanged_idx in range(0, best_unchanged_amount):
        past_generation_networks[best_unchanged_idx].pop(0)
        next_generation_networks.append(past_generation_networks[best_unchanged_idx])


# Metodas apskaiciuoja (iskviecia skaicavimo metoda) kiekvieno tinklo fitness function
# Grazina masyva su fitnessu kiekvienam tinklui atitinkamose masyvo pozicijose
def calculateFitness(networks):
    # !! TUSCIAS METODAS !!
    # Sukurti masyva, tokio dydzio kaip networks, jame ikelti fitnessai, pozicijos masyve tos pacios atitinkamam
    # tinklui networks masyve
    fitness = CNN.runCNN(networks)
    return fitness


# Metodas is dvieju pasirinktu tinklu padaro sukryzminta
def crossover1(parent1, parent2, max_layer_amount, max_conv_depth, max_fc_size, input_length):
    # Crossover1 veikimas- kazkokia dalis parent1 tinklo yra istrinama, i ta vieta yra perkialiama kazkokia
    # parent2 tinklo dalis. Pirmas Conv layeris ir paskutinis FC layeris yra irgi pakeiciami arba suvidurkinami

    # cross_net bus talpinamas naujas netas
    cross_net = parent1

    # Kuriami random range, is kurio is parent 2 bus paimami sluoksniai ir perkeliami i parent 1 sluoksniu vietas
    while True:
        rand_range_1 = []
        rand_range_2 = []

        # Kuriamas range, is kurio bus isimami sluoksniai parent1
        for rand_num in range(0, 2):
            rand_range_1.append(random.randint(1, len(parent1) - 2))
        # Kuriamas range, is kurio bus perkeliami sluoksniai is parent2 i parent1
        for rand_num in range(0, 2):
            rand_range_2.append(random.randint(1, len(parent2) - 2))
        if max(rand_range_2) - min(rand_range_2) - (max(rand_range_1) - min(rand_range_1)) + 2 <= max_layer_amount:
            break

    # Trinami sluoksniai parent1
    for layer_idx in range(min(rand_range_1), max(rand_range_1)):
        cross_net.pop(layer_idx)

    # Ikeliami sluoksniai is parent2 i parent1
    for layer_idx in range(min(rand_range_2), max(rand_range_2)):
        cross_net.insert(min(rand_range_1) + (layer_idx - min(rand_range_2)), parent2[layer_idx])

    crossover_first_layer_type = random.randint(0, 1)
    if crossover_first_layer_type == 0:
        cross_net = crossover_first_last_layers_swap(cross_net, parent1)
    else:
        cross_net = crossover_first_last_layers_merge(cross_net, parent1, parent2, max_conv_depth, max_fc_size,
                                                      input_length)

    # Po crossoverio padaryt metoda, kuris patikrintu, ar visas CNN yra tinkamos strukturos, jei ne tai
    # pakeistu reikiamus kintamuosius taip, kad sutaptu dimensijos
    return cross_net


# Metodas sutvarko neuroninio tinklo struktura. Jei po mutaciju ar kryzminimo neatitinka layerio seka, pvz pooling
# kernelio dydis dalinasi su liekana is inputo dimensiju, arba yra kitokiu layeriu po fc layerio, pertvarkyti
# taip, kad tas nebutu pazeista
def fix_network(network, max_conv_depth, max_fc_size, input_length):
    # Issaugojamas input_length, kuris bus keiciamas sluoksniu tikrinimo eigoje
    cur_input_length = input_length
    # Tikrinimui, ar fc layeris buvo esant kitiems layeriams po jo
    fc_passed = False

    for layer_idx in range(0, len(network) - 1):
        # Jei conv tipo layeris
        if network[layer_idx][0] == "Conv":
            # Jei pries ji buvo FC layeris tai layeri perrasyti random fc layeriu
            if fc_passed:
                network[layer_idx] = startGenerateFCLayer(max_fc_size)
                continue

            # Jei kernelio dydis per mazas arba per didelis
            if network[layer_idx][1] < 1:
                network[layer_idx][1] = 1
            elif network[layer_idx][1] > input_length:
                network[layer_idx] = input_length

            # Jei kerneliu depth per mazas arba per didelis
            if network[layer_idx][2] < 1:
                network[layer_idx][2] = 1
            elif network[layer_idx][2] > max_conv_depth:
                network[layer_idx] = max_conv_depth

        # Jei pool tipo layeris
        elif network[layer_idx][0] == "Pool":
            # Jei pries ji buvo FC layeris tai layeri perrasyti random fc layeriu
            if fc_passed:
                network[layer_idx] = startGenerateFCLayer(max_fc_size)
                continue

            # Jei kSize netinkamas, mazinti ji po viena iki 2 ir ziureti ar tinkamas
            pool_kSize = network[layer_idx][1]
            while cur_input_length % pool_kSize != 0 and pool_kSize > 2:
                pool_kSize -= 1
                if cur_input_length % pool_kSize == 0:
                    network[layer_idx][1] = pool_kSize
                    break

            # Jei kSize vis tiek netinkamas, didinti ji iki cur_input_length - 1 ir ziureti ar tinkamas
            pool_kSize = network[layer_idx][1]
            while cur_input_length % pool_kSize != 0 and pool_kSize < cur_input_length - 1:
                pool_kSize += 1
                if cur_input_length % pool_kSize == 0:
                    network[layer_idx][1] = pool_kSize
                    break

            # Jei kSize vis tiek netinkamas, prilyginti ji 1
            if cur_input_length % network[layer_idx][1] != 0:
                network[layer_idx][1] = 1

            # Pakeisti inputo ilgi sekanciam layeriui
            cur_input_length /= network[layer_idx][1]

        elif network[layer_idx][0] == "FC":
            # Jei fc nodu kiekis mazesnis uz 1, padaryti 1
            if network[layer_idx][1] < 1:
                network[layer_idx][1] = 1

            # Jei fc nodu kiekis didesnis uz limita, padaryti ju skaiciu limito skaiciumi
            elif network[layer_idx][1] > max_fc_size:
                network[layer_idx][1] = max_fc_size

            # Uztvirtinama, kad einant per sluoksnius FC sluoksnis jau buvo
            fc_passed = True

    return network


# Parenkamas "vidurkis" pagal gauso skirstini tarp abieju layeriu parent1 ir parent2
def crossover_first_last_layers_merge(cross_net, parent1, parent2, max_conv_depth, max_fc_size, input_length):
    conv_kSize = 0
    # Jei kernel size didesnis uz inputo dydi ar mazesnis uz 1, generuoti skaiciu is naujo
    while conv_kSize > input_length or conv_kSize < 1:
        rand = random.gauss(0, (max(parent1[0][1], parent2[0][1]) - min(parent1[0][1], parent2[0][1])) / 2)
        conv_kSize = int(min(parent1[0][1], parent2[0][1]) + rand)

    conv_depth = 0
    # Jei kernel depth yra daugiau uz limita ar maziau uz 1, generuoti skaiciu is naujo
    while conv_depth > max_conv_depth or conv_depth < 1:
        rand = random.gauss(0, (max(parent1[0][2], parent2[0][2]) - min(parent1[0][1], parent2[0][1])) / 2)
        conv_depth = int(min(parent1[0][2], parent2[0][2]) + rand)

    fc_size = 0
    # Jei fc nodu kiekis mazesnis uz 1 arba didenis uz limita, generuoti skaiciu is naujo
    while fc_size > max_fc_size or fc_size < 1:
        rand = random.gauss(0, (max(parent1[len(parent1) - 1][2], parent2[len(parent2) - 1][2])
                                - min(parent1[len(parent1) - 1][2],
                                      parent2[len(parent2) - 1][2])) / 2)
        fc_size = min(parent1[len(parent1) - 1][2], parent2[len(parent2) - 1][2]) + rand

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
        cross_net.append(parent2[0])
    return cross_net


def main():
    NETWORKS = startGenerateFromScratch(30, 28, 6, 64, 1000)
    fitness = calculateFitness(NETWORKS[0])
    print(fitness)
    for NET in NETWORKS:
        print(NET)


if __name__ == "__main__":
    main()
