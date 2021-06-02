from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras import backend as K
import saveModel

def evaluateModel(model, X_test, Y_test, batchSize, iters, results_save_path):

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    yp = yp.ravel().astype(int)

    Y_test = np.round(Y_test,0)
    Y_test = Y_test.ravel().astype(int)

    c_matrix = confusion_matrix(Y_test, yp)
    tn, fp, fn, tp = confusion_matrix(Y_test, yp).ravel()
    
    jaccard = dice = ACC = SE = SP = PRE = 0.0

    SE = tp/(tp+fn+1)
    SP = tn/(tn+fp+1)
    PRE = tp/(tp+fp+1)
    ACC = (tn + tp) / (tn + fp + fn + tp)
    jaccard = tp/(tp+fn+fp)
    dice = 2*tp/(2*tp+fn+fp)

    print('Jaccard Index : '+str(jaccard))
    print('Dice Coefficient : '+str(dice))
    print('PRE Coefficient : '+str(PRE))
    print('Accuracy : '+str(ACC))
    print('SE : '+str(SE))
    print('SP : '+str(SP))

    fp = open(results_save_path + '\\jaccard-{}.txt'.format(iters),'a')
    fp.write(str(jaccard)+'\n')
    fp.close()
    fp = open(results_save_path + '\\dice-{}.txt'.format(iters),'a')
    fp.write(str(dice)+'\n')
    fp.close()
    fp = open(results_save_path + '\\pre-{}.txt'.format(iters),'a')
    fp.write(str(PRE)+'\n')
    fp.close()
    fp = open(results_save_path + '\\acc-{}.txt'.format(iters),'a')
    fp.write(str(ACC)+'\n')
    fp.close()
    fp = open(results_save_path + '\\se-{}.txt'.format(iters),'a')
    fp.write(str(SE)+'\n')
    fp.close()
    fp = open(results_save_path + '\\sp-{}.txt'.format(iters),'a')
    fp.write(str(SP)+'\n')
    fp.close()


    fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(jaccard>float(best)):
        print('***********************************************')
        print('Jaccard Index improved from '+str(best)+' to '+str(jaccard))
        fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'w')
        fp.write(str(jaccard))
        fp.close()
        saveModel.saveModel(model, iters, results_save_path)


    fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(dice>float(best)):
        print('***********************************************')
        print('Dice Index improved from '+str(best)+' to '+str(dice))
        fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'w')
        fp.write(str(dice))
        fp.close()

    fp = open(results_save_path + '\\best-pre-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(PRE>float(best)):
        print('***********************************************')
        print('PRE Index improved from '+str(best)+' to '+str(PRE))
        fp = open(results_save_path + '\\best-pre-{}.txt'.format(iters),'w')
        fp.write(str(PRE))
        fp.close()

    fp = open(results_save_path + '\\best-acc-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(ACC>float(best)):
        print('***********************************************')
        print('Accuracy improved from '+str(best)+' to '+str(ACC))
        fp = open(results_save_path + '\\best-acc-{}.txt'.format(iters),'w')
        fp.write(str(ACC))
        fp.close()
    
    fp = open(results_save_path + '\\best-se-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(SE>float(best)):
        print('***********************************************')
        print('SE improved from '+str(best)+' to '+str(SE))
        fp = open(results_save_path + '\\best-se-{}.txt'.format(iters),'w')
        fp.write(str(SE))
        fp.close()
    
    fp = open(results_save_path + '\\best-sp-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(SP>float(best)):
        print('***********************************************')
        print('SP improved from '+str(best)+' to '+str(SP))
        fp = open(results_save_path + '\\best-sp-{}.txt'.format(iters),'w')
        fp.write(str(SP))
        fp.close()

