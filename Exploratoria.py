from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from pylab import rcParams


def plotting_resampled(X_vis, y, X_res_vis, y_resampled, title, uri):
    f, (ax1, ax2) = plt.subplots(1, 2)

    c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Normal",
                     alpha=0.5)
    c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Fraude",
                     alpha=0.5)
    ax1.set_title('Original set')

    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label="Normal", alpha=.5)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label="Fraude", alpha=.5)
    ax2.set_title(title)

    # make nice plotting
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        ax.set_xlim([-20000, -4000])
        ax.set_ylim([-25, 110])

    plt.figlegend((c0, c1), ('Normal', 'Fraude'), loc='lower center',
                  ncol=2, labelspacing=0.)
    plt.tight_layout(pad=3)
    plt.savefig(uri)
    plt.show()


df = pd.read_csv("creditcard.csv")

# Mostrar primeras cinco filas del csv
# df.head(n=5)

# Mostrar dimensiones de los datos
# print(df.shape)

# Mostrar la cantidad de datos por clases
# print(pd.value_counts(df['Class'], sort = True))

# Grafico de barras de las clases
LABELS = ["Normal", "Fraude"]
count_classes = pd.value_counts(df['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.xticks(range(2), LABELS)
plt.title("Frequencia por número de observación")
plt.xlabel("Class")
plt.ylabel("Número de Observaciones")
plt.savefig("PNG/Preprocesamiento/Gráfico de barras de clases Imbalanced.png")
plt.show()

# Dividir los conjuntos de entrenamiento y prueba
X = df.drop(['Class', 'Amount'], axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_train)

# Aplicar UnderSampler a los datos de entrenamiento
#rus = RandomUnderSampler(sampling_strategy='majority')
#X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
#X_test_under, y_test_under = X_test, y_test
#X_under_vis = pca.transform(X_train_under)

#print(X_train_under.shape)
#print(pd.value_counts(y_train_under, sort = True))

# LABELS = ["Normal", "Fraude"]
# count_classes = pd.value_counts(y_train_under, sort = True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.xticks(range(2), LABELS)
# plt.title("Frequencia por número de observación")
# plt.xlabel("Class")
# plt.ylabel("Número de Observaciones")
# plt.savefig("PNG/Preprocesamiento/Gráfico de barras de clases UnderSampler.png")
# plt.show()

# Graficar los datos con UnderSampler
# plotting_resampled(X_vis, y_train, X_under_vis, y_train_under, "UnderSampler", "PNG/Preprocesamiento/Datos UnderSampler.png")

# Aplicar OverSampler a los datos de entrenamiento
#ros = RandomOverSampler(sampling_strategy='minority')
#X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
#X_test_over, y_test_over = X_test, y_test
#X_over_vis = pca.transform(X_train_over)

# print(X_train_over.shape)
# print(pd.value_counts(y_train_over, sort = True))

# LABELS = ["Normal", "Fraude"]
# count_classes = pd.value_counts(y_train_over, sort = True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.xticks(range(2), LABELS)
# plt.title("Frequencia por número de observación")
# plt.xlabel("Class")
# plt.ylabel("Número de Observaciones")
# plt.savefig("PNG/Preprocesamiento/Gráfico de barras de clases OverSampler.png")
# plt.show()

# Graficar los datos con OverSampler
#plotting_resampled(X_vis, y_train, X_over_vis, y_train_over, "OverSampler", "PNG/Preprocesamiento/Datos OverSampler.png")

# Aplicar SMOTE a los datos de entrenamiento
#smote = SMOTE(sampling_strategy='minority')
#X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#X_test_smote, y_test_smote = X_test, y_test
#X_smote_vis = pca.transform(X_train_smote)

# print(X_train_smote.shape)
# print(pd.value_counts(y_train_smote, sort = True))

# LABELS = ["Normal", "Fraude"]
# count_classes = pd.value_counts(y_train_smote, sort = True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.xticks(range(2), LABELS)
# plt.title("Frequencia por número de observación")
# plt.xlabel("Class")
# plt.ylabel("Número de Observaciones")
# plt.savefig("PNG/Preprocesamiento/Gráfico de barras de clases SMOTE.png")
# plt.show()

# Graficar los datos con SMOTE
#plotting_resampled(X_vis, y_train, X_smote_vis, y_train_smote, "SMOTE", "PNG/Preprocesamiento/Datos SMOTE.png")

# Aplicar ADASYN a los datos de entrenamiento
adasyn = ADASYN(sampling_strategy='minority')
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
X_test_adasyn, y_test_adasyn = X_test, y_test
X_adasyn_vis = pca.transform(X_train_adasyn)

# print(X_train_adasyn.shape)
# print(pd.value_counts(y_train_adasyn, sort = True))

# LABELS = ["Normal", "Fraude"]
# count_classes = pd.value_counts(y_train_adasyn, sort = True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.xticks(range(2), LABELS)
# plt.title("Frequencia por número de observación")
# plt.xlabel("Class")
# plt.ylabel("Número de Observaciones")
# plt.savefig("PNG/Preprocesamiento/Gráfico de barras de clases ADASYN.png")
# plt.show()

# Graficar los datos con ADASYN
plotting_resampled(X_vis, y_train, X_adasyn_vis, y_train_adasyn, "ADASYN", "PNG/Preprocesamiento/Datos ADASYN.png")