def saveModel(model, iters, results_save_path, k):
    # model_json = model.to_json()
    # fp = open(results_save_path + '\\modelP.json','w')
    # fp.write(model_json)
    # model.save(results_save_path + '\\model-{}.h5'.format(str(iters)))
    model.save_weights(results_save_path + '\\modelW-{}-{}.h5'.format(str(iters), k))