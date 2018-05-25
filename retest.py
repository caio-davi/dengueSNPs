import _pickle as cPickle

with open('bolstered_DF.pkl', 'rb') as fid:
    bolstered_DF = cPickle.load(fid)
with open('bolstered_FHD.pkl', 'rb') as fid:
    bolstered_FHD = cPickle.load(fid)
with open('mlp_classifier.pkl', 'rb') as fid:
    mlp_classifier = cPickle.load(fid)

tp = fp = tn = fn = 0
predict = mlp_classifier.predict(bolstered_DF)

for i in range(0,len(predict)):
    if(predict[i] == 0):                                    ## Correct Prediction                       
            tn += 1                                         ## True negative
    else:                                                   ## Wrong Prediction
            fp += 1                                         ## False positive

predict = mlp_classifier.predict(bolstered_FHD)
for i in range(0,len(predict)):
    if(predict[i] == 1):                                    ## Correct Prediction                      
            tp += 1                                         ## True positive
    else:                                                   ## Wrong Prediction
            fn += 1                                         ## False negative

total = tp + tn + fp + fn
acc_b = (tp+tn)/total
sen_b = tp/(tp+fn) if (tp+fn)!= 0 else 0
spc_b = tn/(tn+fp) if (tn+fp)!= 0 else 0
err_b = (fp + fn) / total

print("Bolstered Resubstituition Results")
print("bResub ACC: ", acc_b)
print("bResub SEN: ", sen_b)
print("bResub SPC: ", spc_b)
print("bResub ERR: ", err_b)
print("\n")