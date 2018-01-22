import CNNmnist as cnn
import gc

#Apačioje pavyzdys kaip turėtu atrodyti individo arhitektūros formatas:
CA = [['Conv', 15, 60], ['Pool', 4], ['FC', 141, 1], ['FC', 551, 1], ['FC', 963, 1]]
#a = cnn.execCNN(CA)*100
'''
for _ in range(100):
    a = cnn.execCNN(CA)*100
    gc.collect()
    print(a)
'''

def runCNN(CA):
    print(CA)
    rez = cnn.execCNN(CA)*100
    #Šiukšlių surinkėjas kažkodėl neišsprendžia GPU memory problemos
    gc.collect()
    return rez