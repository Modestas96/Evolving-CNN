#Skirtas testavimui

import CNNExecution as CNNexe
import time

CA = [[['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]],
      [['Conv', 5, 64], ['Pool', 2], ['Conv', 5, 64], ['Pool', 2], ['FC', 354, 1]]]


print(CNNexe.CNNExecution().evaluate_population(CA))