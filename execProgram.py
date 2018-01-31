#Skirtas testavimui

import CNNExecution as CNNexe
import time


#[BatchSize]
CA = [

    [['Conv', 12, 26], ['Conv', 8, 40], ['APool', 3], ['FC', 793, 1]],

[['Conv', 12, 26], ['Conv', 8, 40], ['APool', 3], ['FC', 793, 1]],

[['Conv', 12, 26], ['Conv', 8, 40], ['APool', 3], ['FC', 793, 1]]
]

#a1 = CNNexe.CNNExecution(200000).evaluate_population(CA)

r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0
r6 = 0
a2 = CNNexe.CNNExecution(20000).evaluate_population(CA)
print(CA)
for _ in range(10):

    r2 += a2[0]

print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)