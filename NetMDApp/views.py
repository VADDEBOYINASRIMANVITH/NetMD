from django.http import HttpResponse
from .forms import MyfileUploadForm
from .models import file_upload
from django.shortcuts import render, redirect, get_object_or_404
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import json
import argparse
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from pandas_profiling import ProfileReport



def home(request):
    return render(request, 'home1.html')


def ingestion(request):
    all_data = file_upload.objects.all()
    if request.method == 'POST':
        c_form = MyfileUploadForm(request.POST, request.FILES)

        if c_form.is_valid():
            name = c_form.cleaned_data['file_name']
            the_files = c_form.cleaned_data['files_data']
            file_upload(file_name=name, my_file=the_files).save()

            return HttpResponse("File Uploaded Successfully")
        else:
            return HttpResponse("Error in uploading")

    else:
        context = {
            'form': MyfileUploadForm(),
            'data': all_data
        }
        return render(request, 'ingestion.html', context)


def show_file(request):
    all_data = file_upload.objects.all()

    context = {
        'data': all_data
    }
    return render(request, 'view.html', context)


def file_delete(request, id=None):
    instance = get_object_or_404(file_upload, ids=id)
    instance.delete()
    return redirect("view")


def exploratory(request):
    all_data = file_upload.objects.all()
    print(request.method)
    global fname
    if request.method == 'POST':
        fname = request.POST['filename']
        print("gg:", fname)
    global gval

    def gval():
        return fname

    context = {
        'data': all_data
    }
    return render(request, 'exploratory.html', context)


def graph(request):
    print("graphy:", request.method)
    gr = gval()
    print(gr)
    dataset = r'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + gr + '.csv'

    df = pd.read_csv(dataset)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    yres = Counter(y)
    l = len(pd.unique(y))
    x_axis = list(yres.values())
    y_axis = list(yres.keys())
    plt.switch_backend('agg')
    fig, ax = plt.subplots(figsize=(16, 11))

    ax.barh(y_axis, x_axis)

    plt.yticks(np.arange(0, l, 1.0))
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='black')

    ax.set_title("Class distribution Graph of  "+ gr+ " dataset")
    ax.set_xlabel('Class')
    ax.set_ylabel('No. of samples')
    plt.savefig('NetMDApp/static/graphs/' + gr + '.png')
    context = {
        'myname': 'NetMDApp/static/graphs/' + gr + '.png'
    }

    image_data = open(context['myname'], "rb").read()

    graph.n = context['myname']
    return HttpResponse(image_data, content_type="image/png")


# def visualize(request):
#     # data = pd.read_csv('C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/DDOS_dataset.csv')
#     # prof = ProfileReport(data)
#     # prof.to_file(output_file='templates/outputddos.html')
#     return render(request, 'visualize.html')


def preprocessing(request):
    all_data = file_upload.objects.all()
    print(request.method)
    global fname
    if request.method == 'POST':
        fname = request.POST['filename']

    global val

    def val():
        return fname

    context = {
        'data': all_data
    }
    return render(request, 'preprocessing.html', context)


def SB(request):
    selected = val()
    path = r'C:\Users\AJAY THERALA\PycharmProjects\NetMD\media'
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    if request.method == 'POST':
        sb_name = request.POST['sb']
        print(sb_name)
    df = pd.read_csv(dataset)
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    if sb_name == 'Standard Scaler':
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        # col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc)
        fn = selected + '_ss'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == 'Min Max Scaler':
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.transform(X_test)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_mm'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == 'Smote':
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_smote'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == 'Under Sampling':
        nm = NearMiss()
        X_train, y_train = nm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_us'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == 'Over Sampling':
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_os'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == "Standard Scaler and Smote":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_ss_smote'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == "Standard Scaler and Under Sampling":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        nm = NearMiss()
        X_train, y_train = nm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_ss_us'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == "Standard Scaler and Over Sampling":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_ss_os'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == "Min Max Scaler and Smote":
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.transform(X_test)
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_mm_smote'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    elif sb_name == "Min Max Scaler and Under Sampling":
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.transform(X_test)
        nm = NearMiss()
        X_train, y_train = nm.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_mm_us'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    else:
        mm = MinMaxScaler()
        X_train = mm.fit_transform(X_train)
        X_test = mm.transform(X_test)
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_combined_sc = np.r_[X_train, X_test]
        y_combined = np.r_[y_train, y_test]
        data_sc = np.column_stack((X_combined_sc, y_combined))
        col = list(df.columns)
        df1_sc = pd.DataFrame(data_sc, columns=col)
        fn = selected + '_mm_os'
        file_upload(file_name=fn, my_file=df1_sc.to_csv(os.path.join(path, fn + '.csv'))).save()
    return HttpResponse("Successful")


def algo(request):
    all_data = file_upload.objects.all()
    print(request.method)
    global algname
    if request.method == 'POST':
        algname = request.POST['algorithm']
    global value

    def value():
        return algname

    context = {
        'data': all_data
    }
    return render(request, 'algo.html', context)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def LR(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'

    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    data = {'accuracy_lr': metrics.accuracy_score(y_test, y_pred_lr),
            'cm_lr': confusion_matrix(y_test, y_pred_lr),
            'mse_lr': metrics.mean_squared_error(y_test, y_pred_lr),
            'mae_lr': metrics.mean_absolute_error(y_test, y_pred_lr),
            'rmse_lr': np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)),
            'rsquared_lr': metrics.r2_score(y_test, y_pred_lr)
            }

    plot_confusion_matrix(cm_lr, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\lr.png")
    return render(request, 'LR.html', data)


def SVM(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = SVC(kernel='rbf', C=1, gamma=1)
    model.fit(X_train, y_train)
    y_pred_svm = model.predict(X_test)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    Overalldata = {'accuracy_svm': metrics.accuracy_score(y_test, y_pred_svm),
                   'cm_svm': confusion_matrix(y_test, y_pred_svm),
                   'mse_svm': metrics.mean_squared_error(y_test, y_pred_svm),
                   'mae_svm': metrics.mean_absolute_error(y_test, y_pred_svm),
                   'rmse_svm': np.sqrt(metrics.mean_squared_error(y_test, y_pred_svm)),
                   'rsquared_svm': metrics.r2_score(y_test, y_pred_svm)
                   }
    plot_confusion_matrix(cm_svm, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\svm.png")
    return render(request, 'SVM.html', Overalldata)


def RF(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    regressor = RandomForestClassifier(n_estimators=500)
    regressor.fit(X_train, y_train)
    y_pred_rf = regressor.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    data = {'accuracy_rf': metrics.accuracy_score(y_test, y_pred_rf),
            'cm_rf': confusion_matrix(y_test, y_pred_rf),
            'mse_rf': metrics.mean_squared_error(y_test, y_pred_rf),
            'mae_rf': metrics.mean_absolute_error(y_test, y_pred_rf),
            'rmse_rf': np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)),
            'rsquared_rf': metrics.r2_score(y_test, y_pred_rf)
            }
    plot_confusion_matrix(cm_rf, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\rf.png")

    return render(request, 'RF.html', data)


def NB(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred_nb = classifier.predict(X_test)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    data = {'accuracy_nb': metrics.accuracy_score(y_test, y_pred_nb),
            'cm_nb': confusion_matrix(y_test, y_pred_nb),
            'mse_nb': metrics.mean_squared_error(y_test, y_pred_nb),
            'mae_nb': metrics.mean_absolute_error(y_test, y_pred_nb),
            'rmse_nb': np.sqrt(metrics.mean_squared_error(y_test, y_pred_nb)),
            'rsquared_nb': metrics.r2_score(y_test, y_pred_nb)
            }
    plot_confusion_matrix(cm_nb, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\nb.png")

    return render(request, 'NB.html', data)


def CB(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    classifier = CatBoostClassifier()
    classifier.fit(X_train, y_train)
    y_pred_cb = classifier.predict(X_test)
    cm_cb = confusion_matrix(y_test, y_pred_cb)
    data = {'accuracy_cb': metrics.accuracy_score(y_test, y_pred_cb),
            'cm_cb': confusion_matrix(y_test, y_pred_cb),
            'mse_cb': metrics.mean_squared_error(y_test, y_pred_cb),
            'mae_cb': metrics.mean_absolute_error(y_test, y_pred_cb),
            'rmse_cb': np.sqrt(metrics.mean_squared_error(y_test, y_pred_cb)),
            'rsquared_cb': metrics.r2_score(y_test, y_pred_cb)
            }
    plot_confusion_matrix(cm_cb, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\cb.png")

    return render(request, 'CB.html', data)


def AB(request):
    selected = value()
    dataset = 'C:/Users/AJAY THERALA/PycharmProjects/NetMD/media/' + selected + '.csv'
    df = pd.read_csv(dataset)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    l = len(pd.unique(y))
    class_names = list(pd.Series(range(0, l)))
    class_names = [str(s) for s in class_names]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    y_pred_ab = model.predict(X_test)
    cm_ab = confusion_matrix(y_test, y_pred_ab)
    data = {'accuracy_ab': metrics.accuracy_score(y_test, y_pred_ab),
            'cm_ab': confusion_matrix(y_test, y_pred_ab),
            'mse_ab': metrics.mean_squared_error(y_test, y_pred_ab),
            'mae_ab': metrics.mean_absolute_error(y_test, y_pred_ab),
            'rmse_ab': np.sqrt(metrics.mean_squared_error(y_test, y_pred_ab)),
            'rsquared_ab': metrics.r2_score(y_test, y_pred_ab)
            }
    plot_confusion_matrix(cm_ab, classes=class_names, normalize=False, title='Confusion matrix')
    plt.savefig(r"C:\Users\AJAY THERALA\PycharmProjects\NetMD\NetMDApp\static\cm\ab.png")

    return render(request, 'AB.html', data)
