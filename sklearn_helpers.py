#Train a model and apply it to both train and test data
def model_process(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)    
    y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)    
    return y_train_predict, y_test_predict
#Calculating the score of a particular model on both train and test data
def score(model, score, score_name, y_train, y_test, y_train_predict, y_test_predict, verbose = True):
    train_c = score(y_train, y_train_predict)
    test_c = score(y_test, y_test_predict)
    if verbose:
        print(score_name + " score. Train: ", train_c)
        print(score_name + " score. Test: ", test_c)
    return train_c, test_c
#Calculating multiple scores of a particular model on both train and test data
def multi_score(model, score_list, y_train, y_test, y_train_predict, y_test_predict, verbose = True):
    print(str(model))
    score_dic = {'train':{},'test':{}}
    if verbose:
        print('---------------------------')
    is_first = True
    for score_ in score_list:
        if is_first:
            is_first = False
        elif verbose:
            print('---------------------------')
        score_name = str(score_).split()[1]
        score_dic['train'][score_name],score_dic['test'][score_name] = score(model, score_, score_name, y_train, y_test, y_train_predict, y_test_predict, verbose)  
    return score_dic
#Calculating multiple scores of multiple models on both train and test data
def multi_model_multi_score(model_list, score_list, x_train, x_test, y_train, y_test, verbose = True):
    multi_score_dic = {}
    for model in model_list:
        model_name = str(model)
        y_train_predict, y_test_predict = model_process(model, x_train, x_test, y_train)
        multi_score_dic[model] = multi_score(model, score_list, y_train, y_test, y_train_predict, y_test_predict, verbose)
    return multi_score_dic
