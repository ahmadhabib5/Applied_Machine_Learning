#!/usr/bin/env python
# coding: utf-8

# # Recomendation system : Book Recommendation 

# ### Oleh : [Ahmad Habib Husaini](https://www.linkedin.com/in/ahmad-habib-husaini-1705711b0/)
# 
# #### Pendahuluan
# Pada proyek ini, topik yang dibahas adalah mengenai pembuatan sistem rekomendasi dengan data book . Proyek ini dibuat untuk proyek Submission 2 - Machine Learning Terapan Dicoding.

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

# ## 2.1 Data Acquisition

# In[2]:


if os.path.exists('book-recommendation-dataset'):
    print("file sudah ada")
else:
    opendatasets.download_kaggle_dataset(dataset_url='https://www.kaggle.com/arashnic/book-recommendation-dataset', data_dir='')


# Pada tahapan *data loading* berisikan akuisisi data dengan mengunduh dataset pada [link](https://www.kaggle.com/arashnic/book-recommendation-dataset) dengan menggunakan library `opendatasets` dengan sintaks seperti diatas. Ketika merunning code tersebut akan diminta mengisikan username dan key. 
# 1. username silahkan disisi dengan username *account* kaggle
# 2. key didapatkan dengan :
#     1. Buka website [kaggle](https://www.kaggle.com/), 
#     2. login dengan akun masing-masing. 
#     3. Pilih your profile pada kanan atas 
#     4. Pilih *account*
#     5. Scroll sedikit kebawah maka akan ada pilihan `Create new API token`
#     <img src="image/kaggle_token.png" style="zoom:5%;" /> <br>
#     6. Setelah menekan pilihan tersebut akan terunduh file kaggle.json yang berisikan username dan key

# ## 2.2 Memuat data dalam format Dataframe

# ### 2.2.1 Book Dataframe 

# In[3]:


df_book = pd.read_csv('book-recommendation-dataset/Books.csv')
df_book.head(3)


# ### 2.2.2 Rating Dataframe

# In[4]:


df_rating = pd.read_csv('book-recommendation-dataset/Ratings.csv')
df_rating.head(3)


# ### 2.2.4 User Dataframe

# In[5]:


df_user = pd.read_csv('book-recommendation-dataset/Users.csv')
df_user.tail(3)


# # 3. Data Understanding

# ## 3.1 Informasi dataset

# ![image/kaggle_dataset.png](attachment:image.png)<br>
# Pada berkas yang dapat diunduh pada link [berikut](https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Users.csv) berisikan dataset berjumlah 3 buah file yakni Book.csv, Rating.csv dan User.csv
# 

# ### 3.1.1 Book
# 
# Berisikan data sebanyak 271360 baris dan 8 kolom yang terdiri dari: <br>
# 
# Kolom  | Keterangan
# :------------- | :-------------
# Sumber  | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv
# ISBN | International Standard Book Number merupakan kode unik masing masing buku
# Book-title | Merupakan judul dari tiap buku
# Book-Author | Merupakan penulis atau pengarang tiap buku
# Year-Of_Publication |Merupakan tahun terbit tiap buku
# Publisher | Merupakan lebaga penerbit tiap buku
# Image-URL-S | Link foto dari tiap buku berukuran kecil
# Image-URL-M | Link foto dari tiap buku berukuran sedang
# Image-URL-L | Link foto dari tiap buku berukuran besar

# ### 3.1.2 Rating
# Berisikan data sebanyak 1149780 baris dan 3 kolom yang terdiri dari:
# 
# Kolom  | Keterangan
# :------------- | :-------------
# Sumber  | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Ratings.csv
# User-ID | Merupakan kode unik tiap user
# ISBN | International Standard Book Number merupakan kode unik masing masing buku
# Book-Rating |Merupakan rating dari tiap buku

# ### 3.1.3 User
# 
# Berisikan data sebanyak 278858 baris dan 4 kolom yang terdiri dari: <br>
# 
# Kolom  | Keterangan
# :------------- | :-------------
# Sumber  | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Users.csv
# User-ID | Merupakan kode unik tiap user
# Age | Merupakan umur tiap user
# Location |Merupakan lokasi tiap user 

# ## 3.2 Cek Missing Values, tipe data & summary statistical descriptive

# ### 3.2.1 Book

# In[6]:


pd.DataFrame(df_book.isnull().sum(), columns=['missing_values'])


# In[7]:


df_book.info()


# In[8]:


df_book.loc[:,:'Publisher'].describe(include=object)


# ### 3.2.2 Rating

# In[9]:


pd.DataFrame(df_rating.isnull().sum(), columns=['missing_values'])


# In[10]:


df_rating.info()


# In[11]:


df_rating[['Book-Rating']].describe()


# In[12]:


df_rating['Book-Rating'] = df_rating['Book-Rating'].astype(object)


# In[13]:


df_rating.describe(include=[object])


# In[14]:


df_rating.info()


# ### 3.2.3 User

# In[15]:


df_user.head()


# In[16]:


df_user.Location.unique()


# In[17]:


df_user.Location.str.split(',',expand=True).loc[:,:2][0].unique()


# In[18]:


len(df_user.Location.str.split(',',expand=True).loc[:,:2][0].unique())


# In[19]:


df_user.Location.str.split(',',expand=True).loc[:,:2][1].unique()


# In[20]:


len(df_user.Location.str.split(',',expand=True).loc[:,:2][1].unique())


# In[21]:


df_user['city'] = df_user.Location.str.split(',',expand=True)[0]
# df_user['nation_state'] = df_user.Location.str.split(',',expand=True)[1]
df_user['country'] = df_user.Location.str.split(',',expand=True)[2]
df_user.head()


# In[22]:


pd.DataFrame(df_user.isnull().sum(), columns=['missing_values'])


# In[23]:


df_user.info()


# In[24]:


pd.set_option('display.max_rows',105)


# In[25]:


df_user.groupby('country')[['country']].count().head(10)


# In[26]:


df_user.drop(columns=['country'], inplace=True)


# In[27]:


df_user[['Age']].describe()


# In[28]:


df_user.describe(include=object)


# `Kesimpulan dari Tahapan ini :`
# 1. Book
#     - Buku dengan judul terbanyak adalah Selected Poems dengan jumlah 27 buah
#     - Penulis terbanyak adalah Agatha Christie dengan total 632 karya
#     - Buku terbitan tahun 2002 menjadi buku terbanyak
#     - Lembaga penerbit yang paling banayk menerbitkan buku adalah Harlequin
# 2. Rating
#     - masih banyak buku yang belum dirating, yakni sebanyak 716109
# 3. User
#     - Banyak data umur yang kosong kemungkinan tidak akan digunakan
#     - london menjadi kota user terbanyak sebanyak 139187 user

# ## 3.3 Visualisasi Data

# ### 3.3.1 Book

# In[29]:


count_book = {key: len(df_book[key].unique()) for key in df_book.columns[1:5]}


# In[30]:


plt.figure(figsize=(15,9))
ax = sns.barplot(x=list(count_book.keys()), y=list(count_book.values()))
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_title("Banyak Data Unik Tiap Kolom Dataset Book")
plt.show()


# ### 3.3.2 Rating

# In[31]:


count_rating = {key: len(df_rating[key].unique()) for key in df_rating.columns}


# In[32]:


plt.figure(figsize=(15,9))
ax = sns.barplot(x=list(count_rating.keys()), y=list(count_rating.values()))
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_title("Banyak Data Unik Tiap Kolom Dataset Rating")
plt.show()


# In[33]:


plt.figure(figsize=(15,9))
ax = sns.countplot(data=df_rating, x='Book-Rating')
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_title("Banyak Data Unik Kolom Rating")
plt.show()


# ### 3.3.3 Users

# In[34]:


df_user.head()


# In[35]:


count_users = {key: len(df_user[key].unique()) for key in df_user.columns}


# In[36]:


plt.figure(figsize=(12,9))
ax = sns.barplot(x=list(count_users.keys()), y=list(count_users.values()))
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_title("Banyak Data Unik Tiap Kolom Dataset User")
plt.show()


# In[37]:


city_top10 = df_user.city.value_counts().head(10).reset_index()
city_top10.columns = ['city', 'count']
city_top10


# In[38]:


plt.figure(figsize=(12,9))
ax = sns.barplot(x=city_top10.city, y=city_top10['count'])
for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax.set_title("Top 10 Kota Terbanyak")
plt.show()


# # 4. Data Preparation & Data Preprocessing

# ## 4.1 Memilih fitur untuk sistem rekomendasi

# In[39]:


df_fix = df_book.iloc[:,0:4]
df_fix.head()


# In[40]:


pd.DataFrame({
    'missing_values':df_fix.isnull().sum(),
    'presentase':df_fix.isnull().sum()/len(df_fix)
})


# ## 4.2 Hapus Missing Value dan data duplicate pada dataset fix

# In[41]:


df_fix.dropna(inplace=True)
df_fix.drop_duplicates(inplace=True)


# In[42]:


df_fix.shape


# ## 4.3 Create Metadata

# In[43]:


df_fix['Year-Of-Publication'] = df_fix['Year-Of-Publication'].astype('str')


# In[44]:


df_fix['metadata'] = ""
for col in ['Book-Title', 'Book-Author', 'Year-Of-Publication']:
    df_fix['metadata'] += df_fix[col]
    df_fix['metadata'] += " "


# In[45]:


df_fix.tail()


# ## 4.4 Encoding dengan TF-IDF

# In[46]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[47]:


bow = CountVectorizer()
bank = bow.fit_transform(df_fix.metadata)


# In[48]:


tfidf = TfidfVectorizer()
bank2 = tfidf.fit_transform(df_fix.metadata)


# # 5. Modeling

# ## 5.1 Encoding metadata index

# In[49]:


index = 0
content = df_fix.loc[index, 'metadata']
content


# In[50]:


code1 = bow.transform([content])
code1


# In[51]:


code2 = tfidf.transform([content])


# ## 5.2 Document Search

# In[52]:


from sklearn.metrics.pairwise import cosine_distances


# In[53]:


distance1 = cosine_distances(code1, bank)
distance1


# In[54]:


rec_idx = distance1.argsort()[0, 1:]
rec_idx


# In[55]:


df_fix.loc[rec_idx[:5], :]


# In[56]:


distance2 = cosine_distances(code2, bank)
distance2


# In[57]:


rec_idx2 = distance2.argsort()[0, 1:]
rec_idx2


# In[58]:


df_fix.loc[rec_idx2[:10], :]


# ## 5.3 Bungkus code agar rapih

# In[59]:


class RecommendationSystems():
    def __init__(self,data, metadata_col):
        self.df = data
        self.metadata = metadata_col
        self.encoder = None
        self.bank = None
    
    def fit(self, encoder='tfidf'):
        if encoder not in ['bow', 'tfidf']:
            print("hanya support BoW dan TF-IDF")
        else:
            self.encoder = TfidfVectorizer()
            if encoder=='bow':
                self.encoder = CountVectorizer()
            self.bank = self.encoder.fit_transform(self.df[self.metadata])
    def recommender(self, index, topn=10):
        content = self.df.loc[index, self.metadata]
        code = self.encoder.transform([content])
        distance = cosine_distances(code, self.bank)
        rec_idx = distance.argsort()[0, 1:]
        return self.df.loc[rec_idx[:topn], :]


# In[60]:


recsys = RecommendationSystems(df_fix, 'metadata')
recsys.fit()


# In[61]:


rec = recsys.recommender(100)
rec


# In[62]:


df_fix.loc[100,:]


# In[63]:


recsys2 = RecommendationSystems(df_fix, 'metadata')
recsys2.fit('bow')


# In[64]:


rec2 = recsys2.recommender(100)
rec2


# ## 5.4 Scoring

# ## 5.4.1 Try

# In[65]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[66]:


book = pd.DataFrame(dict(df_fix.loc[100,:]), index=[0])
book


# In[67]:


rec2


# In[68]:


np.mean(cosine_similarity(bow.transform(book.metadata.values), bow.transform(rec.metadata)))


# In[69]:


np.mean(cosine_similarity(tfidf.transform(book.metadata.values), tfidf.transform(rec.metadata)))


# ## 5.4.2 Bungkus kembali kedalam class

# In[70]:


class RecommendationSystems():
    def __init__(self,data, metadata_col):
        self.df = data
        self.metadata = metadata_col
        self.encoder = None
        self.bank = None
        self.content = None
        self.index = None
        self.topn = None
    
    def fit(self, encoder='tfidf'):
        if encoder not in ['bow', 'tfidf']:
            print("hanya support BoW dan TF-IDF")
        else:
            self.encoder = TfidfVectorizer()
            if encoder=='bow':
                self.encoder = CountVectorizer()
            self.bank = self.encoder.fit_transform(self.df[self.metadata])
    def recommender(self, index, topn=10, include_book_search=False):
        self.index = index       
        self.topn = topn
        self.content = self.df.loc[index, self.metadata]
        code = self.encoder.transform([self.content])
        distance = cosine_distances(code, self.bank)
        rec_idx = distance.argsort()[0, :]
        if include_book_search:
            topn+=1
            return self.df.loc[rec_idx[:topn], :] 
        return self.df.loc[rec_idx[1:topn], :]

    def score(self):
        rec = self.recommender(index=self.index, topn=self.topn, include_book_search=True)
        bank = self.encoder.transform(rec.metadata)
        code = self.encoder.transform([self.content])
        return np.mean(cosine_similarity(code, bank)[:,1:])


# In[71]:


recsys3 = RecommendationSystems(df_fix, 'metadata')
recsys3.fit()
recsys3.recommender(index=100, include_book_search=True)


# In[72]:


recsys3.score()


# In[73]:


import time


# In[74]:


start = time.time()
recsys4 = RecommendationSystems(df_fix, 'metadata')
recsys4.fit('bow')
recsys4.recommender(index=100)
stop = time.time()
print(f"Training time: {stop - start} s")
print("Score : ",recsys4.score())


# ## 5.4.3 Scoring Model dengan *Bag Of Word* top 5 rekomendasi

# In[75]:


recsys = RecommendationSystems(data=df_fix, metadata_col='metadata')
recsys.fit(encoder='bow')


# In[76]:


recsys.recommender(index=100, topn=5, include_book_search=True)


# In[77]:


recsys.score()


# In[78]:


recsys.recommender(index=1000, topn=5, include_book_search=True)


# In[79]:


recsys.score()


# ## 5.4.4 Scoring Model Tf-idf

# In[80]:


recsys = RecommendationSystems(data=df_fix, metadata_col='metadata')
recsys.fit(encoder='tfidf')


# In[81]:


recsys.recommender(index=100, topn=5, include_book_search=True)


# In[82]:


recsys.score()


# In[83]:


recsys.recommender(index=1000, topn=5, include_book_search=True)


# In[84]:


recsys.score()


# # 6 Evaluasi

# ## 6.1 Generate data yang akan dijadikan sampel sebanyak 500 data

# In[85]:


np.random.randint(0,10, 5)


# In[86]:


idx_sample_list = np.random.randint(0,len(df_fix), 500)
idx_sample_list


# In[87]:


len(idx_sample_list)


# In[88]:


idx_sample_list2 = [i for i in np.random.randint(0,len(df_fix), 500) if i not in idx_sample_list]
len(idx_sample_list2)


# In[89]:


from tqdm import tqdm


# ## 6.2 Define Loop Function

# In[90]:


def loop_rec(data, metadata_col='metadata', encoder='tfidf', sample_list=[0,2,10],topn=10, return_dict=False):
    score_list = []
    score_dict = {}
    idx_done = []
    recsys = RecommendationSystems(data, metadata_col)
    recsys.fit(encoder)
    for i in tqdm(sample_list):
        while True:
            idx = np.random.randint(low=0, high=len(data))
            if idx not in idx_done:
                idx_done.append(idx)
                break
        recsys.recommender(index=idx, topn=topn)
        score = recsys.score()
        score_list.append(score)
        score_dict[idx] = score
    if return_dict:
        return score_dict, np.mean(score_list)
    return np.mean(score_list)


# ## 6.3 Model Bag-of-word 

# In[91]:


dict_score_bow, score_bow = loop_rec(data=df_fix,
                                     metadata_col='metadata',
                                     encoder='bow',
                                     topn=5,
                                     sample_list=idx_sample_list,
                                     return_dict=True)


# In[92]:


score_bow


# In[93]:


dict_score_bow2, score_bow2 = loop_rec(data=df_fix,
                                     metadata_col='metadata',
                                     encoder='bow',
                                     topn=5,
                                     sample_list=idx_sample_list2,
                                     return_dict=True)


# In[94]:


score_bow2


# ## 6.4 Model Tf-idf

# In[95]:


dict_score_tfidf, score_tfidf = loop_rec(data=df_fix,
                                     metadata_col='metadata',
                                     encoder='tfidf',
                                     topn=5,
                                     sample_list=idx_sample_list,
                                     return_dict=True)


# In[96]:


score_tfidf


# In[97]:


dict_score_tfidf2, score_tfidf2 = loop_rec(data=df_fix,
                                     metadata_col='metadata',
                                     encoder='tfidf',
                                     topn=5,
                                     sample_list=idx_sample_list2,
                                     return_dict=True)


# In[98]:


score_tfidf2


# In[100]:


pd.DataFrame({
    'Tf-Idf':[score_tfidf, score_tfidf2],
    'BoW':[score_bow, score_bow2]
})


# In[ ]:




