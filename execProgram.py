#Skirtas testavimui

import CNNExecution as CNNexe
import time

#[BatchSize]
CA = [
      [['Conv', 5, 64], ['MPool', 2], ['Conv', 5, 64], ['MPool', 2], ['FC', 1024, 1]]
]

#a1 = CNNexe.CNNExecution(200000).evaluate_population(CA)

r1 = 0
r2 = 0
r3 = 0
r4 = 0
r5 = 0
r6 = 0
for _ in range(10):

    a2 = CNNexe.CNNExecution(20000).evaluate_population(CA)
    r2 += a2[0]

print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)