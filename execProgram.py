#Skirtas testavimui

import CNNExecution as CNNexe
import time
import Genetic as ge

#[BatchSize]
CA = [
    [['Conv', 5, 20], ['Conv', 7, 20], ['Conv', 2, 36], ['Conv', 8, 22], ['APool', 3], ['FC', 994, 0.7666]],
    [['Conv', 7, 20], ['Conv', 3, 35], ['Conv', 8, 22], ['Conv', 8, 22], ['APool', 3], ['FC', 974, 0.7666]],
    [['Conv', 5, 31], ['Conv', 8, 22], ['Conv', 5, 31], ['Conv', 2, 36], ['Conv', 8, 22], ['APool', 3], ['FC', 974, 0.7666]],
    [['Conv', 7, 20], ['Conv', 3, 35], ['Conv', 8, 22], ['APool', 3], ['FC', 974, 0.7666]],
    [['Conv', 7, 20], ['Conv', 2, 62], ['Conv', 7, 20], ['Conv', 2, 36], ['Conv', 8, 22], ['APool', 3], ['FC', 974, 0.7666]],
    [['Conv', 7, 20], ['Conv', 7, 20], ['Conv', 2, 36], ['Conv', 9, 31], ['APool', 3], ['FC', 974, 0.7666]],


    [['Conv', 4, 26], ['Conv', 4, 26], ['Conv', 5, 52], ['MPool', 3], ['FC', 871, 0.68962]],
    [['Conv', 4, 39], ['Conv', 4, 26], ['Conv', 4, 26], ['Conv', 5, 52], ['MPool', 3], ['Conv', 3, 38], ['FC', 952, 0.88563]],
    [['Conv', 4, 26], ['Conv', 4, 26], ['Conv', 5, 52], ['MPool', 3], ['Conv', 3, 38], ['FC', 952, 0.88563]],
    [['Conv', 4, 26], ['Conv', 5, 52], ['MPool', 3], ['Conv', 3, 38], ['FC', 952, 0.88563]],
    [['Conv', 4, 26], ['Conv', 5, 52], ['MPool', 3], ['Conv', 2, 42], ['Conv', 3, 38], ['FC', 878, 0.68962]],
    [['Conv', 4, 26], ['Conv', 5, 30], ['Conv', 5, 52], ['Conv', 6, 29], ['MPool', 3], ['FC', 736, 0.68962]],


    [['Conv', 4, 56], ['Conv', 4, 56], ['Conv', 4, 56], ['MPool', 2], ['MPool', 2], ['Conv', 1, 10], ['Conv', 5, 21], ['FC', 977, 0.69569]],
    [['Conv', 4, 56], ['Conv', 4, 56], ['Conv', 4, 56], ['MPool', 2], ['APool', 2], ['Conv', 5, 21], ['FC', 977, 0.69569]],
    [['Conv', 4, 56], ['Conv', 4, 56], ['Conv', 4, 56], ['MPool', 2], ['Conv', 1, 10], ['Conv', 5, 21], ['FC', 959, 0.69569]],
    [['Conv', 4, 56], ['Conv', 4, 56], ['MPool', 2], ['Conv', 4, 56], ['Conv', 1, 10], ['Conv', 5, 21], ['FC', 959, 0.69569]],
    [['Conv', 4, 56], ['Conv', 4, 56], ['MPool', 2], ['Conv', 4, 56], ['MPool', 2], ['Conv', 1, 10], ['Conv', 5, 21], ['FC', 959, 0.69569]],
    [['Conv', 4, 56], ['Conv', 4, 63], ['Conv', 4, 56], ['MPool', 2], ['Conv', 1, 10], ['Conv', 6, 61], ['FC', 888, 0.69569]],


    [['Conv', 8, 12], ['Conv', 9, 53], ['APool', 2], ['Conv', 8, 48], ['APool', 2], ['FC', 916, 0.77023]],
    [['Conv', 8, 12], ['Conv', 9, 53], ['APool', 2], ['Conv', 8, 48], ['APool', 3], ['FC', 983, 0.77023]],
    [['Conv', 8, 12], ['Conv', 9, 53], ['Conv', 8, 48], ['APool', 3], ['FC', 983, 0.77023]],
    [['Conv', 8, 12], ['Conv', 8, 12], ['Conv', 8, 54], ['APool', 2], ['Conv', 8, 48], ['APool', 2], ['FC', 910, 0.77023]],
    [['Conv', 8, 11], ['Conv', 9, 53], ['APool', 2], ['Conv', 8, 48], ['APool', 2], ['FC', 983, 0.77023]],
    [['Conv', 8, 12], ['Conv', 8, 12], ['Conv', 9, 53], ['APool', 2], ['Conv', 8, 48], ['APool', 2], ['FC', 983, 0.77023]],


    [['Conv', 9, 21], ['Conv', 3, 38], ['Conv', 9, 21], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]],
    [['Conv', 8, 21], ['Conv', 3, 38], ['Conv', 3, 38], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]],
    [['Conv', 9, 21], ['Conv', 3, 38], ['Conv', 3, 38], ['Conv', 9, 21], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]],
    [['Conv', 11, 21], ['Conv', 3, 38], ['Conv', 3, 38], ['Conv', 11, 21], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]],
    [['Conv', 11, 21], ['Conv', 3, 38], ['Conv', 3, 38], ['Conv', 3, 38], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]],
    [['Conv', 9, 21], ['Conv', 3, 38], ['Conv', 3, 38], ['Conv', 11, 21], ['APool', 3], ['FC', 843, 0.86514], ['FC', 843, 0.86514]]
    
]

#a1 = CNNexe.CNNExecution(200000).evaluate_population(CA)

r1 = []
ab = []

for i in range(len(CA)):
    ab.append(0)


for _ in range(1):
    a2 = CNNexe.CNNExecution(220000).evaluate_population(CA, False)
    for i in range(len(CA)):
        ab[i] += a2[i]
    print(ab)

ge.write_debug_to_file(CA, -1, ab)
print(ab)