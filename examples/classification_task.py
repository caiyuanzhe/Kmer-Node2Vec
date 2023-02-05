import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable
import logging
import warnings
warnings.filterwarnings("ignore") 


logger = logging.getLogger()
logging.basicConfig(filename=f"../data_dir/input/precision_dataset/4g/classification_txtfile/log/classification_Equal_quantity.log", filemode='a',
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S",
                    level=logging.DEBUG)

# set your own file path
def path(method):
    if method == "3x6000_kmernode2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/kmernode2vec_8mer_fungi_6000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/kmernode2vec_8mer_bacteria_6000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/kmernode2vec_8mer_viral_6000Vectors.txt"

    elif method == "3x6000_dna2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/dna2vec_8mer_fungi_6000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/dna2vec_8mer_bacteria_6000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x6000/dna2vec_8mer_viral_6000Vectors.txt"
    
    elif method == "3x7000_kmernode2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/kmernode2vec_8mer_fungi_7000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/kmernode2vec_8mer_bacteria_7000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/kmernode2vec_8mer_viral_7000Vectors.txt"

    elif method == "3x7000_dna2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/dna2vec_8mer_fungi_7000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/dna2vec_8mer_bacteria_7000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x7000/dna2vec_8mer_viral_7000Vectors.txt"

    elif method == "3x8000_kmernode2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/kmernode2vec_8mer_fungi_8000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/kmernode2vec_8mer_bacteria_8000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/kmernode2vec_8mer_viral_8000Vectors.txt"

    elif method == "3x8000_dna2vec-8mer":
        path_fungi = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/dna2vec_8mer_fungi_8000Vectors.txt"
        path_bacteria = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/dna2vec_8mer_bacteria_8000Vectors.txt"
        path_viral = "../data_dir/input/precision_dataset/4g/classification_txtfile/3x8000/dna2vec_8mer_viral_8000Vectors.txt"

    else:
        return 0

    logging.info(f"--------------{method}--------------")
    print(f"--------------{method}--------------")
    return path_fungi,path_bacteria,path_viral


def concatenate(path_fungi,path_bacteria,path_viral):
    data_fungi = np.loadtxt(path_fungi)
    data_bacteria = np.loadtxt(path_bacteria)
    data_viral = np.loadtxt(path_viral)

    X = np.concatenate((data_fungi, data_bacteria, data_viral), axis=0)
    X = pd.DataFrame(X)

    # concat Y，data_fungi, data_bacteria, data_viral whose label is 0，1，2, respectively
    Y0 = np.zeros(data_fungi.shape[0])
    Y1 = np.ones(data_bacteria.shape[0])
    Y2 = np.ones(data_viral.shape[0])
    Y2 = Y2 + 1.0
    Y = np.concatenate((Y0, Y1, Y2), axis=0)

    Y = pd.DataFrame(Y)
    Y.astype(int)

    return X,Y


def train_multi(X,Y,k):
    model = RandomForestClassifier()

    # concat the table to print results
    A=["accuracy"]
    B=["precision_macro"]
    C=["recall_macro"]
    D=["f1_macro"]
    E = ["specificaty"]
    headers = ['evaluation', "RandomForest"]  


    y_pred = cross_val_predict(model, X, Y.values.ravel(), cv=k)
    Accuracy = round(accuracy_score(Y.values.ravel(), y_pred),3) #round()保留三位小数
    Precision = round(precision_score(Y.values.ravel(), y_pred, average='macro'),3)
    Recall = round(recall_score(Y.values.ravel(), y_pred, average='macro'),3)
    F1 = round(f1_score(Y.values.ravel(), y_pred, average='macro'),3) 

    # tn, fp, fn, tp = confusion_matrix(Y.values.ravel(), y_pred).ravel()
    # Specificity = tn / (tn + fp)
    spe = []

    # confusion_matrix
    con_mat = confusion_matrix(Y.values.ravel(), y_pred)
    for i in range(3):  # 3 denotes three labels
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    Specificity = round(sum(spe) / len(spe),3)

    A.append(Accuracy)
    B.append(Precision)
    C.append(Recall)
    D.append(F1)
    E.append(Specificity)

    table = PrettyTable()
    table.field_names = headers
    table.add_row(A)
    table.add_row(B)
    table.add_row(C)
    table.add_row(D)
    table.add_row(E)

    logging.info(table)
    print(table)
    logging.info('')
    print(" ")


def loop(method,k):
    logging.info(f"{k}fold")
    print(f"{k}fold")
    for i in method:
        path_fungi, path_bacteria, path_viral = path(i)
        x_data, y_data = concatenate(path_fungi, path_bacteria, path_viral)
        train_multi(x_data, y_data,k)


if __name__ == '__main__':
    method = ["3x6000_kmernode2vec-8mer","3x6000_dna2vec-8mer","3x7000_kmernode2vec-8mer","3x7000_dna2vec-8mer","3x8000_kmernode2vec-8mer","3x8000_dna2vec-8mer"]
    k=10 # 10fold-cross validation
    loop(method,k)
