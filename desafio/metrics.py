import pandas as pd
import argparse

class Metrics:
    def __init__(self, threshold = 0.5):
        self.threshold = threshold
    
    def pos_neg(self, data):
        TP = TN = FP = FN = 0
        for idx in list(data.index.values):
            pred = 1 if data['predicted'][idx] > self.threshold else 0

            if pred == 1 and data['target'][idx] == 1:
                TP += 1
            elif pred == 0 and data['target'][idx] == 0:
                TN += 1
            elif pred == 1 and data['target'][idx] == 0:
                FP += 1
            else:
                FN += 1
        return TP, TN, FP, FN
        
    def calculate_metrics(self, data):
        APCER = 0
        for s in range(1, 11):
            aux_data = data[data['spoof_type'] == s]
            _, TN, FP, _ = self.pos_neg(aux_data)
            if FP != 0:
                aux_APCER = FP / (TN + FP)
                if aux_APCER > APCER:
                    APCER = aux_APCER
        
        TP, TN, FP, FN = self.pos_neg(data)
        
        NPCER = FN / (FN + TP)
        ACER  = (APCER + NPCER) / 2
        FPR   = FP / (FP + TN)
        TPR   = TP / (TP + FN)
        PPV   = TP / (TP + FP)
        F1    = 2 * ((TPR * PPV) / (TPR + PPV))

        return round(APCER, 3),  round(NPCER, 3), round(ACER, 3), round(FPR, 3), round(TPR, 3), round(PPV, 3), round(F1, 3)

    def accuracy(self, data):
        TP, TN, FP, FN = self.pos_neg(data)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        
        return round(ACC, 3)
        

parser = argparse.ArgumentParser(add_help = True, description = 'Metrics for Face Anti-Spoofing')
parser.add_argument('--path', type = str, default = './predictions/001_64_50.csv', help='Path to csv')
parser.add_argument('--threshold', type = float, default = 0.5, help='Threshold')
args = parser.parse_args()

filename = args.path.split('/')
filename = filename[-1][:-4]+'_'+str(args.threshold)+'.txt'

data = pd.read_csv(args.path)
m = Metrics(args.threshold)

APCER, NPCER, ACER, FPR, TPR, PPV, F1 = m.calculate_metrics(data)

with open('./results/'+filename, 'a') as f:
    f.write('APCER: '+str(APCER)+'\n')
    f.write('NPCER: '+str(NPCER)+'\n')
    f.write('ACER: '+str(ACER)+'\n')
    f.write('FPR: '+str(FPR)+'\n')
    f.write('TPR: '+str(TPR)+'\n')
    f.write('PPV: '+str(PPV)+'\n')
    f.write('F1: '+str(F1)+'\n')
    f.close()

for s in range(11):
    aux_data = data[data['spoof_type'] == s]
    ACC = m.accuracy(aux_data)
    with open('./results/'+filename, 'a') as f:
        f.write('ACC spoof_type '+str(s)+': '+str(ACC)+'\n')
        f.close()
    
for i in range(5):
    aux_data = data[data['illumination'] == i]
    ACC = m.accuracy(aux_data)
    with open('./results/'+filename, 'a') as f:
        f.write('ACC illumination '+str(i)+': '+str(ACC)+'\n')
        f.close()
    
for e in range(3):
    aux_data = data[data['environment'] == e]
    ACC = m.accuracy(aux_data)
    with open('./results/'+filename, 'a') as f:
        f.write('ACC environment '+str(e)+': '+str(ACC)+'\n')
        f.close()