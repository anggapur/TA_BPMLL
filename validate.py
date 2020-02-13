pred = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
test = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

def intersections(pred,test):
    hasil = []
    for index, data in enumerate(pred):
        if (data == 1 and test[index] == 1):
            hasil.append(1)
        else:
            hasil.append(0)
    return hasil

def unions(pred,test):
    hasil = []
    for index, data in enumerate(pred):
        if (data == 1 or test[index] == 1):
            hasil.append(1)
        else:
            hasil.append(0)
    return hasil

def validation(pred,test):
    data_len = len(pred);
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f_measure = 0.0
    for ix,x in enumerate(pred):
        predict = pred[ix]
        # print(predict) # yang di predict
        correct = test[ix]
        # print(correct) # yang correct

        P = sum(predict)
        C = sum(correct)

        CnP = float(sum((intersections(pred[ix],test[ix]))))
        CuP = float(sum((unions(pred[ix], test[ix]))))

        # print((intersections(pred[ix],test[ix])))
        # print((unions(pred[ix], test[ix])))
        #
        # print(CnP)
        # print(CuP)


        if (CuP == 0):
            accuracy += 0
        else:
            accuracy += (CnP / CuP);

        if (P == 0):
            precision += 0
        else:
            precision += (CnP / P)

        if (C == 0):
            recall += 0
        else:
            recall += (CnP / C)

        if ((C + P) == 0):
            f_measure += 0
        else:
            f_measure += (2 * (CnP) / (C + P))

        # print("---------")
        # print("Acc Sum : " + str(accuracy));

    accuracy = accuracy / data_len
    # print("Acc : "+str(accuracy));

    # print("Precision Sum : " + str(precision));
    precision = precision / data_len
    # print("Precision : " + str(precision));

    # print("Recall Sum : " + str(recall));
    recall = recall / data_len
    # print("Recall : " + str(recall));

    # print("F Measure Sum : " + str(f_measure));
    f_measure = f_measure / data_len
    # print("F Measure : " + str(f_measure));

    data_hasil = (str(accuracy)+","+str(precision)+","+str(recall)+","+str(f_measure))
    return data_hasil


validation(pred,test)

