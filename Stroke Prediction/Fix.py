#!/usr/bin/env python
# coding: utf-8

# # Analysis Predictive: Prediction positive or negative stroke

# ### Oleh : [Ahmad Habib Husaini](https://www.linkedin.com/in/ahmad-habib-husaini-1705711b0/)
# 
# #### Pendahuluan
# Pada proyek ini, topik yang dibahas adalah mengenai kesehatan yang di buat untuk memprediksi pasien apakah menderita diabeste atau tidak. Proyek ini dibuat untuk proyek Submission 1 - Machine Learning Terapan Dicoding.

# # 1. Import important package

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import opendatasets
import wget
import zipfile
from tqdm import tqdm
import os


# # 2. Data loading

# ### 2.1 Data Acquisition

# In[2]:


if os.path.exists('stroke-prediction-dataset/healthcare-dataset-stroke-data.csv'):
    print("file sudah ada")
else:
    opendatasets.download_kaggle_dataset(dataset_url='https://www.kaggle.com/fedesoriano/stroke-prediction-dataset', data_dir='')


# Pada tahapan *data loading* berisikan akuisisi data dengan mengunduh dataset pada [link](https://www.kaggle.com/fedesoriano/stroke-prediction-datase)dengan menggunakan library `opendatasets` dengan sintaks seperti diatas. Ketika merunning code tersebut akan diminta mengisikan username dan key. 
# 1. username silahkan disisi dengan username *account* kaggle
# 2. key didapatkan dengan :
#     1. Buka website [kaggle](https://www.kaggle.com/), 
#     2. login dengan akun masing-masing. 
#     3. Pilih your profile pada kanan atas 
#     4. Pilih *account*
#     5. Scroll sedikit kebawah maka akan ada pilihan `Create new API token`
#     <img src="image/kaggle_1_1.png" style="zoom:10%;" /> <br>
#     6. Setelah menekan pilihan tersebut akan terunduh file kaggle.json yang berisikan username dan key

# ### 2.2 Memuat data dalam format Dataframe

# In[3]:


df = pd.read_csv("stroke-prediction-dataset/healthcare-dataset-stroke-data.csv")
df.head()


# In[4]:


df.stroke.unique()


# # 3. Data Understanding

# # informasi data set : 
# Attribute  | Keterangan
# :------------- | :-------------
# Sumber  | https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
# ID | Nomor identitas pasien
# Gender | Jenis kelamin (male, female, other(**dianggap invalid data**))
# Hypertension | Nol jika pasien tidak menderita darah tinggi, satu sebaliknya
# Heart diseease | Nol jika pasien tidak menderita penyakit jantung, satu sebaliknya
# Ever married | Status pernah menikah (Yes or No)
# Work type | Jenis pekerjaan terdiri dari anak-anak, Pekerjaan pemerintah, Tidak pernah bekerja, Swasta atau Wiraswasta
# Residence type | Tipe tempat tinggal Rural (pedasaan) atau Urban (perkotaan)
# Avg glucose type | rata-rata kadar gula dalam darah
# BMI | body mass index
# Smoking status | formerly smoked (Sebelumnya merokok), never smoked (tidak pernah merokok), smokes (merokok) atau "Unknown" (dianggap invalid data)
# Stroke | satu jika pasien positif stroke nol sebaliknya

# ### 3.1 Melihat banyak data

# In[5]:


print("Jumlah baris          :", df.shape[0])
print("Jumlah kolom          :", df.shape[1])
print("Jumlah missing values :", df.isnull().sum().sum())


# `Terdapat 5110 baris, 12 kolom dan 201 missing values.` <br>
# `Kolom apa saja yang terdapat missing values ?` 

# ### 3.2 Cek missing values

# In[6]:


pd.DataFrame({
    'missing value':df.isnull().sum()
})


# `kolom tersebut akan diisikan dengan rata-rata, namum prosesnya akan dilakukan belakangan dengan bantuan pipeline sklearn`

# ### 3.3 cek tipe data tiap kolom

# In[7]:


df.info()


# `Jika dilihat terdapat 7 kolom numerik dan 5 kolom categorical`,`namun jika diperhatikan kolom hypertension dan heart_disease merupakan target yang merupakan data cagetorical, oleh karenanya kita perlu mengubah tipe data dari kolom tersebut`

# ### 3.4 Mengubah tipe data

# In[8]:


df.hypertension = df.hypertension.astype(object)
df.heart_disease = df.hypertension.astype(object)


# In[9]:


df.info()


# ### 3.5 Summary statistical descriptive

# `Numerical`
# 1. Count  adalah jumlah sampel pada data.
# 2. Mean adalah nilai rata-rata.
# 3. Std adalah standar deviasi.
# 4. Min yaitu nilai minimum setiap kolom. 
# 5. 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
# 6. 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
# 7. 75% adalah kuartil ketiga.
# 8. Max adalah nilai maksimum.
# 
# `categorical:`
# 1. unique adalah banyaknya kategori dari setiap kolom categorical
# 2. top adalah kategori paling banyak dari setiap kolom
# 3. freq adalah banyaknya frequensi dari top

# In[10]:


df.describe()


# `pada kolom age nilai min adalah 0.08, pertanyaanya apa maksud dari 0.08 ? `, `ini menunjukan terdapat indikasi invalid data`

# In[11]:


df.describe(include=object)


# `Pada categorical columns tidak terlihat ada yang aneh`

# ### 3.6 Visualisasi data

# ### 3.6.1 Memisahkan fitur kategorik, numerik dan target

# In[12]:


target = ['stroke']
num_feature = ['age', 'avg_glucose_level','bmi']
cat_feature = [i for i in df.columns if i not in (target + num_feature + ['id'])]


# ### 3.6.2 Visualisasi fitur kategorik

# In[13]:


fig, ax = plt.subplots(len(cat_feature),1, figsize=(30,100))
idx=0
for ft in cat_feature:
    sns.countplot(data=df, x=ft, ax=ax[idx])
    for p in ax[idx].patches:
        ax[idx].annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[idx].set_xticklabels(ax[idx].get_xticklabels(), fontsize=13)
    ax[idx].set_title("frequency of "+str(ax[idx].get_xlabel()), fontsize=20)
    ax[idx].set_xlabel("", fontsize=15)
    ax[idx].set_yticklabels(ax[idx].get_yticklabels(), fontsize=13)
    idx+=1
plt.show()


# In[14]:


fig, ax = plt.subplots(len(cat_feature),1, figsize=(30,100))
idx=0
for ft in cat_feature:
    sns.countplot(data=df, x=ft, ax=ax[idx], hue='stroke')
    for p in ax[idx].patches:
        ax[idx].annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    ax[idx].set_xticklabels(ax[idx].get_xticklabels(), fontsize=13)
    ax[idx].set_title("frequency of "+str(ax[idx].get_xlabel()+" against the target"), fontsize=20)
#     ax[idx].set_xlabel(ax[idx].get_xlabel(), fontsize=15)
    ax[idx].set_xlabel("", fontsize=15)
    ax[idx].set_yticklabels(ax[idx].get_yticklabels(), fontsize=13)
    idx+=1
plt.show()


# ### 3.6.3 Visualisasi fitur numerik

# In[15]:


fig, ax = plt.subplots(len(num_feature),1, figsize=(25,30))
idx = 0
for feature in num_feature:
    sns.boxplot(data=df, y=feature, ax=ax[idx])
    ax[idx].set_title("boxplot of "+str(feature), fontsize=20)
    idx+=1


# In[16]:


fig, ax = plt.subplots(len(num_feature),1, figsize=(25,40))
idx=0
for ft in num_feature:
    sns.distplot(df[ft], bins=10, ax=ax[idx])
    ax[idx].set_xticklabels(ax[idx].get_xticks(), fontsize=15)
    ax[idx].set_title("Distibution of "+str(ax[idx].get_xlabel()), fontsize=20)
    ax[idx].set_xlabel("", fontsize=15)
    idx+=1
plt.show()


# ### 3.6.4 Visualisasi Variabel target

# In[17]:


plt.figure(figsize=(15,7))
ax = sns.countplot(data=df, x='stroke')
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_xlabel("", fontsize=15)
ax.set_title("frequent of target")
plt.show()


# ### 3.6.5 visualisasi missing value

# In[18]:


df_pos_null = df[df['stroke']==1].isnull().sum()[:-1].reset_index()
df_pos_null.columns = ["kolom","missing_value"]
df_pos_null


# In[19]:


plt.figure(figsize=(20,5))
sns.barplot(data=df_pos_null, x='kolom', y='missing_value')
plt.title("Missing values in positive stroke")
plt.show()


# In[20]:


df_neg_null = df[df['stroke']==0].isnull().sum()[:-1].reset_index()
df_neg_null.columns = ["kolom","missing_value"]
df_neg_null


# In[21]:


plt.figure(figsize=(20,5))
sns.barplot(data=df_neg_null, x='kolom', y='missing_value')
plt.title("Missing values in negative stroke")
plt.show()


# `Kesimpulan yang dapat diambil :`
# 1. Data target tidak seimbang antara pasien positif dan negatif
# 2. Data dengan value gender other dianggap **invalid data** 
# 3. Data dengan value smoking_status unknown akan dihapus
# 4. Pada kolom numerik terdapat cukup banyak outlier
# 5. Kolom numerik masih condong atau skew

# # 4. Data Preparation

# ### 4.1  Handling invalid data

# `Kolom ID hanyalah nomer unik dari masing-masing pasien`,`kolo tersebut sangat kecil bahkan tidak berpengaruh sama sekali pada target`

# `Pada kolom gender dan smoking status terdapat data invalid`

# In[22]:


df_backup = df.copy(deep=True)


# In[23]:


df[(df.gender == 'Other')]


# In[24]:


df.groupby(by=[cat_feature[0],'stroke'])['id'].count().to_frame()


# In[25]:


df = df_backup.copy(deep=True)


# In[26]:


df = df[(df.gender != 'Other')]
df.shape


# In[27]:


df.groupby(by=[cat_feature[0],'stroke'])['id'].count().to_frame()


# In[28]:


df.groupby(by=[cat_feature[6],'stroke'])['id'].count().to_frame()


# `Jika diihat ternyata pada kolom smoking status = unknown value stroke = 1 ada 47, jika kita hapus tidak terlalu banyak sample stroke = 1 yang hilang`

# In[29]:


df = df[(df.smoking_status !='Unknown')]
df.shape


# In[30]:


df.groupby(by=[cat_feature[6],'stroke'])['id'].count().to_frame()


# ### 4.2 Handling Outlier

# In[31]:


fig, ax = plt.subplots(len(num_feature),1, figsize=(25,30))
idx = 0
for feature in num_feature:
    sns.boxplot(data=df, y=feature, ax=ax[idx])
    ax[idx].set_title(feature)
    idx+=1


# `Jika dilihat dari kolom numeric diatas terdapat cukup banyak outlier`<br>
# `Hal pertama yang perlu dilakukan adalah membuat batas bawah dan batas atas.`<br>
# `Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR.` <br>
# `Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3.`

# In[32]:


Q1 = df[num_feature].quantile(0.25)
Q3 = df[num_feature].quantile(0.75)
IQR = Q3 - Q1
IQR.to_frame()


# `Syarat bukan outlier, (data < (Q1-1.5*IQR)) atau (data > (Q3+1.5*IQR))`

# In[33]:


batas_bawah = Q1 - 1.5*IQR
batas_atas = Q3 + 1.5*IQR


# In[34]:


df = df[~((df < batas_bawah) | (df > batas_atas)).any(axis=1)]
df.shape


# In[35]:


fig, ax = plt.subplots(len(num_feature),1, figsize=(25,30))
idx = 0
for feature in num_feature:
    sns.boxplot(data=df, y=feature, ax=ax[idx])
    ax[idx].set_title(feature)
    idx+=1


# In[36]:


df.stroke.unique()


# In[37]:


fig, ax = plt.subplots(len(num_feature),1, figsize=(25,40))
idx=0
for ft in num_feature:
    sns.distplot(df[ft], bins=10, ax=ax[idx])
    ax[idx].set_xticklabels(ax[idx].get_xticks(), fontsize=15)
    ax[idx].set_title("distibution of "+str(ax[idx].get_xlabel()), fontsize=20)
    ax[idx].set_xlabel("", fontsize=15)
    idx+=1
plt.show()


# `Jika dilihat dari distribusi, kolom sudah hampir bersidtribusi normal dan tidak terlalu condong atau skew`,`Tetapi dalam pipeline nanti tetap akan di transform dengan yeo-johnson`

# ### 4.3 Split Data

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


df[cat_feature] = df[cat_feature].astype(object)


# In[40]:


X = df.drop(columns="stroke")
y = df.stroke

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### 4.4 Build Pipeline

# In[41]:


import sklearn
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder ,MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ### 4.4.1 Numerical Pipeline

# In[42]:


sklearn.set_config(display='diagram')
num_pipline = Pipeline([
    ('inputer', SimpleImputer(strategy='median')), # Handling missing values
    ('transformer', PowerTransformer('yeo-johnson')), # Tranformasi agar berdistribusi normal
    ('scaling', RobustScaler()) # Penyekalaan data
])


# ### 4.4.2 Categorical Pipeline

# In[43]:


cat_pipeline = Pipeline([
    ('encoding', OrdinalEncoder())
])


# ### 4.4.3 Create Preprocessor

# In[44]:


preprocessor = ColumnTransformer([
    ('numeric', num_pipline, num_feature), 
    ('categorical', cat_pipeline, cat_feature)
])


# In[45]:


preprocessor


# `Tahapan diatas untuk menentukan kolom mana yang akan diterapkan numerical pipeline atau categorical pipeline`

# In[47]:


from jcopml.tuning.space import Real, Integer # Library buatan mas WiraDKP https://www.linkedin.com/in/wiradkputra/
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


# # 5. Modeling

# ### 5.1 Model without resampling

# ### 5.1.1 Final pipeline

# In[48]:


pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])


# In[49]:


pipeline


# In[50]:


params_rf = {
    'algo__n_estimators': Integer(low=100, high=200),
    'algo__max_depth':Integer(low=20, high=80),
    'algo__max_features': Real(low=0.1, high=1, prior='uniform'),
    'algo__min_samples_leaf':Integer(low=1, high=20)
}


# In[51]:


model_rf = RandomizedSearchCV(pipeline, params_rf, cv=3, n_iter=50,scoring='f1', n_jobs=-1, verbose=1, random_state=42)
model_rf.fit(X_train, y_train)

print(model_rf.best_params_)
print(model_rf.score(X_train, y_train), model_rf.best_score_, model_rf.score(X_test, y_test))


# ### 5.1.2 Cek confusion matrix model

# In[52]:


from sklearn.metrics import classification_report, confusion_matrix


# In[53]:


cf_matrix = confusion_matrix(y_test, model_rf.predict(X_test))
cf_matrix


# In[54]:


import seaborn as sns
ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.show()


# In[55]:


print(classification_report(y_test, model_rf.predict(X_test)))


# In[56]:


y_test.value_counts()


# `Jika hanya berpacu pada accuracy model memiliki accuracy yang sangat baik, tetapi jika dilihat ternyata model salah semua dalam memprediksi pasien positif`,`oleh karenanya coba lakukan pembobotan dan gunakan scoring f1-score`

# ### 5.2 Model dengan pembobotan

# In[57]:


[{0: x, 1: 1-x} for x in [0.05, 0.1, 0.25]]


# In[58]:


params_rf = {
    'algo__n_estimators': Integer(low=100, high=200),
    'algo__max_depth':Integer(low=20, high=80),
    'algo__max_features': Real(low=0.1, high=1, prior='uniform'),
    'algo__min_samples_leaf':Integer(low=1, high=20),
    'algo__class_weight':[{0: x, 1: 1-x} for x in [0.05, 0.1, 0.25]]
}


# In[59]:


pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])
pipeline


# In[60]:


model_rf_2 = RandomizedSearchCV(pipeline, params_rf, cv=3, n_iter=50, scoring='f1', n_jobs=-1, verbose=1, random_state=42)
model_rf_2.fit(X_train, y_train)

print(model_rf_2.best_params_)
print(model_rf_2.score(X_train, y_train), model_rf_2.best_score_, model_rf_2.score(X_test, y_test))


# In[61]:


cf_matrix = confusion_matrix(y_test, model_rf_2.predict(X_test))
cf_matrix


# In[62]:


ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.show()


# In[63]:


print(classification_report(y_test, model_rf_2.predict(X_test)))


# `Dengan melakukan pembobotan prediksi pada data text menjadi lebih baik walau masih banyak salah prediksi`

# ### 5.3 Resampling data dengan SMOTE

# `Teknik resampling merupakan pembuatan data dummy dengan algoritma tertentu, pada proyek kali ini akan digunakan algoritma SMOTE yang berdasarkan pada algoritma KNN`

# In[64]:


import imblearn, sklearn
from imblearn.over_sampling import SMOTE
print(imblearn.__version__, sklearn.__version__)


# `Resampling dilakukan data train saja, mengapa ? jika data test ikut diresampling dan misalkan accuracy model bagus apa maknanya ? bisa saja data test yang benar diprediksi adalah data buatan hasil resampling, oleh karenanya untuk menilai model baik atau tidak uji dengan data yang asli`

# ### 5.3.1 New Pipeline 

# In[65]:


res_pipeline = Pipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy='minority')),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])
res_pipeline


# ### 5.3.2 Model tanpa tunning parameter Smote

# In[66]:


model_rf_3_1 = RandomizedSearchCV(res_pipeline, params_rf, cv=3, n_iter=50, scoring='f1', n_jobs=-1, verbose=1, random_state=42)
model_rf_3_1.fit(X_train, y_train)

print(model_rf_3_1.best_params_)
print(model_rf_3_1.score(X_train, y_train), model_rf_3_1.best_score_, model_rf_3_1.score(X_test, y_test))


# In[67]:


cf_matrix = confusion_matrix(y_test, model_rf_3_1.predict(X_test))
cf_matrix


# In[68]:


ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.show()


# In[69]:


print(classification_report(y_test, model_rf_3_1.predict(X_test)))


# `Walaupun accuracy berkurang, tetapi model dapat memprediksi pasien positif lebih baik dari sebelumnya`

# ### 5.3.3 Model dengan melakukan tunning parameter smote

# In[70]:


params_rf = {
    'smote__k_neighbors': Integer(low=1, high=50),
    'algo__n_estimators': Integer(low=100, high=200),
    'algo__max_depth':Integer(low=20, high=80),
    'algo__max_features': Real(low=0.1, high=1, prior='uniform'),
    'algo__min_samples_leaf':Integer(low=1, high=20),
    'algo__class_weight':[{0: x, 1: 1-x} for x in [0.05, 0.1, 0.25]]
}


# In[71]:


model_rf_3_2 = RandomizedSearchCV(res_pipeline, params_rf, cv=5, n_iter=50, scoring='f1', n_jobs=-1, verbose=1, random_state=42)
model_rf_3_2.fit(X_train, y_train)

print(model_rf_3_2.best_params_)
print(model_rf_3_2.score(X_train, y_train), model_rf_3_2.best_score_, model_rf_3_2.score(X_test, y_test))


# In[72]:


cf_matrix = confusion_matrix(y_test, model_rf_3_2.predict(X_test))
cf_matrix


# In[73]:


ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.show()


# In[74]:


print(classification_report(y_test, model_rf_3_2.predict(X_test)))


# `Jika diperhatikan dari rumus dibawah ini :`<br> <br>
# $$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$$ <br>
# $$ Precision = \frac{TP}{TP+FP} $$<br>
# $$ Recall = \frac{TP}{TP+FN}$$ <br>
# $$ F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN} $$ <br> <br>
# `Untuk mengecilkan false negatif (salah dalam memprediksi pasien positif, metrics yang cocok adalah recall`

# In[75]:


model_rf_4 = RandomizedSearchCV(res_pipeline, params_rf, cv=5, n_iter=50, scoring='recall', n_jobs=-1, verbose=1, random_state=42)
model_rf_4.fit(X_train, y_train)

print(model_rf_4.best_params_)
print(model_rf_4.score(X_train, y_train), model_rf_4.best_score_, model_rf_4.score(X_test, y_test))


# In[76]:


cf_matrix = confusion_matrix(y_test, model_rf_4.predict(X_test))
cf_matrix


# In[77]:


ax = sns.heatmap(cf_matrix, annot=True,cmap='Blues', fmt='g')
ax.set_xticklabels(["Negatif", "Positif"])
ax.set_yticklabels(["Negatif", "Positif"])
plt.show()


# In[78]:


from sklearn.metrics import f1_score, recall_score


# # Evaluasi

# In[79]:


df_model = pd.DataFrame(columns=['model_1', 'model_2', 'model_3_1', 'model_3_2', 'model_4'], index=['f1_score','recall'])
df_model


# In[80]:


f1_score(y_test, model_rf.predict(X_test))


# In[81]:


models = {
    'model_1':model_rf,
    'model_2':model_rf_2,
    'model_3_1':model_rf_3_1,
    'model_3_2':model_rf_3_2,
    'model_4':model_rf_4
}
metrics = {
    'f1_score':f1_score,
    'recall':recall_score
}
for metric in metrics.keys():
    for model in models.keys():
        df_model.loc[metric, model] = metrics[metric](y_test, models[model].predict(X_test))


# In[82]:


df_model


# `Tetapi hal tersebut menyebabkan banyak prediksi yang salah dari kelas negatif.` `Jika model salah dalam memprediksi pasien negatif (aslinya negatif diprediksi positif) tentu jauh lebih baik ketimbang aslinya positif tetapi diprediksi negatif. Namun kembali lagi pada keputusan klien atau pihak berkepentingan`

# `Dan jika ingin memprebaiki kualitas model maka perbanyak sample dan `**`jangan imbalance`**
