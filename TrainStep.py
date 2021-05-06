import evaluate
import evaluateo
import evaluateiou

def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize, iters, results_save_path, losshistory, reverse = False):

    
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)
        evaluate.evaluateModel(model,X_test, Y_test, batchSize, iters, results_save_path)

    return model