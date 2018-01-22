import CNNmnist as cnn
import gc

#Apačioje pavyzdys kaip turėtu atrodyti individo arhitektūros formatas:
CA = [['Conv', 5, 63], ['Conv', 5, 63], ['Pool', 2], ['Conv', 5, 63], ['FC', 300, 1]]
for _ in range(60):
    cnn.execCNN(CA)*100

def runCNN(CA):
    print(CA)
    rez = cnn.execCNN(CA)*100
    #Šiukšlių surinkėjas kažkodėl neišsprendžia GPU memory problemos
    gc.collect()
    return rez