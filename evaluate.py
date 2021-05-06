import numpy as np
import saveModel

def evaluateModel(model, X_test, Y_test, batchSize, iters, results_save_path):

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

    yp = np.round(yp,0)

    jaccard = dice = ACC = SE = SP = PRE = 0.0
    
    for i in range(len(Y_test)):
        tp = tn = fp = fn = acc = 0
        
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        for i in range(len(y2)):
            if y2[i] == 0:
                if yp_2[i] ==0:
                    tn += 1
                    acc += 1
                else:
                    fp += 1
            else:
                if yp_2[i] ==0:
                    fn += 1
                else:
                    tp += 1
                    acc += 1
        
        
        se = tp/(tp+fn+1)
        SE = SE +se

        sp = tn/(tn+fp+1)
        SP = SP + sp

        pre = tp/(tp+fp+1)
        PRE = PRE + pre
        
        acc = acc / len(y2)
        ACC = ACC + acc

        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jaccard += (np.sum(intersection)/np.sum(union))  
        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))


    jaccard /= len(Y_test)
    dice /= len(Y_test)
    PRE /= len(Y_test)
    ACC /= len(Y_test)
    SE /= len(Y_test)
    SP /= len(Y_test)
    
    
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
