import CNNmnist as cnn
import gc

#Apačioje pavyzdys kaip turėtu atrodyti individo arhitektūros formatas:
CA = [['Conv', 5, 63], ['Pool', 3], ['Pool', 2], ['FC', 100, 1]]
#cnn.execCNN(CA)*100

def runCNN(CA):
    print(CA)
    rez = cnn.execCNN(CA)*100
    #Šiukšlių surinkėjas kažkodėl neišsprendžia GPU memory problemos
    gc.collect()
    return rez