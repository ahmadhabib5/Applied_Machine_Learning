# Laporan Proyek Machine Learning - Ahmad Habib Husaini

## Daftar Isi
- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Project Overview

Buku merupakan salah satu sumber informasi yang sangat diperlukan dalam menambah wawasan serta pengetahuan dimulai dari  *sains*, sosial, budaya, ekonomi  hingga politik. Buku dapat berupa buku fisik maupun buku elektronik. Dengan membaca buku dapat menjadikan pribadi seseorang semakin baik dan menurunkan efek negatif seperti kenakalan pada anak dan semisalnya. Namun menurut UNESCO minat baca buku masyarakat indonesia  sangat rendah yakni hanya 0.001 persen. hal tersebut dapat diartikan dari 1000 orang indonesia hanya 1 yang suka membaca buku. Hal tersebut sangat disayangkan, mengingat jumlah penduduk indonesia yang besar tentunya memiliki potensi untuk menjadi negara yang maju kalau penduduknya rajin membaca buku baik buku fisik atau buku elektronik yang dapat ditemui di perpustakaan, maupun aplikasi  perpustakaan elektronik. 

Pada perpustakaan fisik, buku sudah dikelompokkan berdasarkan kemiripan topik oleh petugas. Bagaimana dengan aplikasi perpustakaan elektronik ?, aplikasi dapat memanfaatkan kecerdasan buatan untuk memberikan rekomendasi berdasarkan kemiripan buku seperti judul, penulis hingga tahun terbit yang dikenal dengan istilah *recommendation system*. Pada proyek kali ini akan dibuat sebuah model *machine learning* berupa *recommendation system* buku yang diharapkan dapat memberikan rekomendasi berupa buku-buku serupa dengan yang dibaca oleh pembaca.  

## Business Understanding
### Problem Statement
Berdasarkan latar belakang yang telah dipaparkan, berikut rincian masalah yang akan disolusikan pada proyek ini:
- Bagaimana cara membuat sistem rekomendasi berdasarkan kemiripan atribut buku ?

### Goals
Berdasarkan *problem statement* diatas, berikut tujuan dari proyek ini:

- Membuat sistem rekomendasi berdasarkan kemiripan  atribut pada buku.

### Solution Approach

Berikut tahapan pada proyek ini dalam membuat *recommendation system* pada buku berdasarkan atribut:

<img src="D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\alur.png" style="zoom: 80%;" />

1. Data Acquisition

   Tahap ini merupakan tahap pengumpulan data dari sumber data. 

2. Data Understanding

   Tahapan ini merupakan tahapan untuk memahami data yang telah dikumpulkan dengan cara mengetahui *summary statistical descriptive* maupun visualisasi.

3. Data Preprocessing (Data Preparation)

   Tahapan ini terdiri dari beberapa tahapan yakni, 

   1. Memilih dataset serta fitur atau kolom untuk dijadikan atribut dari buku.

   3. Menghapus nilai kosong serta data duplikat.

   3. Menggabungkan atribut yang sudah dipilih ke dalam satu kolom berupa String yang dipisahkan dengan spasi.

   4. Melakukan encoding yakni mengubah atau merepresentasikan fitur gabungan tadi menjadi *vector* angka dengan dua pendekatan yakni Bag-of-word dan TF-idf untuk membuat bank data dari kolom metadata. Bag-of-word merupakan salah satu cara encoding pada data teks dengan cara pertama kumpulkan data unik dari tiap teks, Kemudian membuat *vector* dari masing masing teks berdasarkan banyaknya kemunculan, untuk lebih jelasnya bisa melihat ilustrasi dibawah ini.

      | No   | Teks                                 |
      | ---- | ------------------------------------ |
      | 1    | dia belajar machine learning         |
      | 2    | machine learning adalah subdomain AI |
      | 3    | saya belajar deep learning           |
   
      | No   | Corpus (kata unik)                                           |
      | :--- | :----------------------------------------------------------- |
      | 1    | ['AI',  'adalah',  'belajar',  'deep',  'dia',  'learning',  'machine',  'saya',  'subdomain'] |
   
      | No Teks | AI   | adalah | deep | dia  | learning | machine | saya | subdomain |
      | ------- | ---- | ------ | ---- | ---- | -------- | ------- | ---- | --------- |
      | 1       | 0    | 0      | 0    | 1    | 1        | 1       | 0    | 0         |
      | 2       | 1    | 1      | 0    | 0    | 1        | 1       | 0    | 1         |
      | 3       | 0    | 0      | 1    | 0    | 1        | 0       | 1    | 0         |
   
      Maka teks "dia belajar machine learning" setelah di *encode* dengan *bag of word* menjadi [0,0,0,1,1,1,0,0]. 
   
      Sama halnya dengan *bag of word* *tfi-df* juga merupakan salah satu *encoder* untuk merepresentasikan teks menjadi *vector* angka. *Tf-idf* sendiri merupakan pengembangan dari *bag of word* dengan memberikan bobot berdasarkan *inverse document frequency* (IDF). Untuk lebih jelasnya dapat melihat ilustrasi dibawah ini.
   
      | No Teks | AI   | adalah | deep | dia  | learning | machine | saya | subdomain |
      | ------- | ---- | ------ | ---- | ---- | -------- | ------- | ---- | --------- |
      | 1       | 0    | 0      | 0    | 1    | 1        | 1       | 0    | 0         |
      | 2       | 1    | 1      | 0    | 0    | 1        | 1       | 0    | 1         |
      | 3       | 0    | 0      | 1    | 0    | 1        | 0       | 1    | 0         |
      | **DF**  | 1/3  | 1/3    | 1/3  | 1/3  | 3/3      | 2/3     | 1/3  | 1/3       |
      | **IDF** | 3    | 3      | 3    | 3    | 1        | 3/2     | 3    | 3         |
   
      *Tf-idf* adalah dengan mengalikan *vector* hasil *bag of word* dengan nilai *inverse document frequency*. Maka teks "dia belajar *machine learning*" setelah di *encode* dengan *tf-idf* menjadi [0,0,0,1/3,1/3,1/3,0,0] . Namun pada prakteknya tidak menggunakan nilai IDF langsung melainkan len(IDF) + 1.
   
   5. Membuat bank data
   
      Bank data berisikan hasil encoding dari seluruh data berdasarkan kolom gabungan yang telah dibuat.


4. Modeling

   Pada tahapan ini proses pembuatan *recommendation system* dibuat dengan memanfaatkan *cosine similarity*. *Cosine similarity* merupakan salah satu konsep dalam aljabar linier untuk menghitung cosinus dari sudut antara dua *vector*, namun dapat digunakan untuk mengukur kemiripan pada suatu dokumen dan salah satunya seperti pada proyek ini menghitung kemiripan buku berdasarkan atributnya. Berikut formula *cosine similarity*.

   <img src="D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\cosine similarity.png" style="zoom:80%;" />

   Sebagai contoh mengukur kemiripan teks satu dan dua yang telah di *encode* dengan *bag of word*, 

   ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\cosine similarity example.png)

   Rentang nilai dari *cosine similarity* sama dengan rentang nilai cosinus yakni dari -1 hingga 1. Jika nilai *cosine similarity* mendekati dengan nol maka dapat dikatakan kedua dokumen tersebut sangat mirip dan jika nilai dari *cosine similarity* semakin jauh dari nol entah itu mendekat ke 1 atau -1 maka kedua dokumen tersebut dapat dikatakan tidak mirip. Untuk penjelasan lebih lanjut akan dibahas pada tahapan *modeling*

5. Evaluasi

   Pada tahapan evaluasi bertujuan untuk mengukur performa hasil dari sistem rekomendasi yang telah dibuat. *Metric* yang digunakan adalah *intra similarity*, *metric* ini hanyalah rata-rata dari *cosine similarity* antara dokumen asli dengan hasil rekomendasi. Untuk lebih jelasnya akan dipaparkan di tahap [evaluasi](#Evaluation).

## Data Understanding

![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\kaggle_dataset.png)

Pada berkas yang diunduh pada tautan [berikut](https://www.kaggle.com/arashnic/book-recommendation-dataset) berisikan tiga dataset yakni dataset [*book*](https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv) dengan jumlah data 271360 baris dan 8 kolom, dataset [*rating*](https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Ratings.csv) dengan jumlah data 1149780 baris dan 3 kolom dan dataset [*user*](https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Users.csv) dengan jumlah data 278858 baris dan 4 kolom. Berikut rincian data pada masing-masing dataset:

### Dataset Book

- #### Informasi Kolom Dataset 

  | Kolom               | Keterangan                                                   |
  | :------------------ | :----------------------------------------------------------- |
  | Sumber              | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv |
  | ISBN                | International Standard Book Number merupakan kode unik masing masing buku |
  | Book-title          | Merupakan judul dari tiap buku                               |
  | Book-Author         | Merupakan penulis atau pengarang tiap buku                   |
  | Year-Of_Publication | Merupakan tahun terbit tiap buku                             |
  | Publisher           | Merupakan lembaga penerbit tiap buku                         |
  | Image-URL-S         | Link foto dari tiap buku berukuran kecil                     |
  | Image-URL-M         | Link foto dari tiap buku berukuran sedang                    |
  | Image-URL-L         | Link foto dari tiap buku berukuran besar                     |

- #### Informasi Missing Value

  ![](image\missing_value_book.png)

- #### Informasi Tipe Data

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\tipe data book.png)

- #### Informasi Statistical 

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\statistical_book.png)



### Dataset Rating

- #### Informasi Kolom Dataset

  | Kolom       | Keterangan                                                   |
  | :---------- | :----------------------------------------------------------- |
  | Sumber      | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Ratings.csv |
  | User-ID     | Merupakan kode unik tiap user                                |
  | ISBN        | International Standard Book Number merupakan kode unik masing masing buku |
  | Book-Rating | Merupakan rating dari tiap buku                              |

- #### Informasi Missing Value

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\missing_valu_rating.png)

- #### Informasi Tipe Data

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\tipe_data_rating.png)

- #### Informasi Statistical 

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\statistical_rating.png)

### Dataset User

- #### Informasi Kolom Dataset

  | Kolom    | Keterangan                                                   |
  | :------- | :----------------------------------------------------------- |
  | Sumber   | https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Users.csv |
  | User-ID  | Merupakan kode unik tiap user                                |
  | Age      | Merupakan umur tiap user                                     |
  | Location | Merupakan lokasi tiap user dengan nilai ibu kota, negara bagian dan negara serikat |

- #### Informasi Missing Value

  ![image-20211106111804739](C:\Users\WIN10\AppData\Roaming\Typora\typora-user-images\image-20211106111804739.png)

- #### Informasi Tipe Data

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\tipe_data_user.png)

- #### Informasi Statistical 


Berikut `visualisasi` dari ketiga dataset tersebut:

![](image\visualisasi_book.png)

![](image\visualisasi_rating1.png)

![](image\visualisasi_rating2.png)

![](image\visualisasi_user1.png)

![](image\visualisasi_user2.png)

Kesimpulan yang dapat diambil pada tahapan ini adalah :
1. Book
    - Buku dengan judul terbanyak adalah Selected Poems dengan jumlah 27 buah
    - Penulis terbanyak adalah Agatha Christie dengan total 632 karya
    - Buku terbitan tahun 2002 menjadi buku terbanyak
    - Lembaga penerbit yang paling banyak menerbitkan buku adalah Harlequin
2. Rating
    - masih banyak buku yang belum dirating, yakni sebanyak 716109
3. User
    - Banyak data umur yang kosong kemungkinan tidak akan digunakan
    - london menjadi kota user terbanyak sebanyak 139187 user
    - Kota dengan user terbanyak adalah London


## Data Preparation

Seperti yang sudah dipaparkan pada *solution approach*, berikut rincian dari masing-masing tahapan:

- Pada proyek ini seperti yang sudah dipaparkan pada tahapan *business understanding* bahwa *recommendation systems* dibuat berdasarkan kemiripan atribut dari buku, oleh karenanya dataset yang digunakan adalah dataset buku (book.csv). Pada dataset buku kolom yang dipilih untuk dijadikan sebagai atribut adalah ISBN, judul, penulis , dan tahun terbit dari buku. Berikut dataset yang akan digunakan:

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\dataset_fix.png)

- Setelah dipilih beberapa kolom untuk dijadikan atribut, cek apakah dataset yang akan digunakan memiliki *missing value*.

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\missing_value_dataset_fix.png)

  Karena *missing value* hanya sedikit yakni 1 maka tidak mengapa untuk menghapus data tersebut. Kemudian selain menghapus data kosong hapus juga data duplikat. Untuk implementasi kedalam *code* dapat dilihat pada gambar dibawah ini.

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\drop_missing_value_duplicate.png)

- Setelah membersihkan dengan menghapus data kosong atau *missing value* maka kolom yang dipilih untuk dijadikan atribut akan digabungkan menjadi satu kolom berupa string yang dipisahkan dengan spasi, kolom tersebut akan diberi nama 'metadata'. Sebagai contoh data pertama maka metadatanya adalah "0195153448 Classical Mythology Mark P.P. Morford 2002". Untuk implementasinya sangat mudah, dapat dilihat pada gambar dibawah ini.

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\metadata.png)

  Selanjutnya kolom metadata tersebutlah yang akan digunakan untuk membuat *recommendation system*.

- Encoding kolom metadata.

  Pada tahapan ini dilakukan dua pendekatan encoding yakni dengan *bag of word* dan *tf-idf* yang penjelasannya sudah dipaparkan pada tahapan **solution* approach*.


## Modeling

Setela melakukan tahapan preparation atau preprocessing maka tahapan *modeling* data dilakukan. Pada tahapan *solution approach* telah dibahas cara mengukur kemiripan antar dua dokumen, berikut tahapan dari proses *modeling* pada proyek ini dalam membuat *recommendation system*.

1. Memilih index

   Tahapan pertama memilih data berdasarkan index untuk diberikan rekomendasi buku yang mirip.

2. Mencari dokumen berdasarkan *cosine distance*.

   cosine distance merupakan cara mengukur jarak dari sebuah dokumen dengan bantuan cosine similarity. Formula dari *cosine distance* adalah 1 dikurang nilai cosine similarity. Sebagai contoh berdasarkan teks satu dan teks dua yang telah dihitung pada tahapan *solution approach*, maka *cosine distance* dari kedua teks tersebut adalah 1 - 0.5163977794943222 yakni 0.4836022205056778. Semakin besar nilai *cosine distance* semakin tidak mirip kedua dokumen atau teks tersebut.

3. Mengurutkan dokumen dengan *cosine distance* dari paling kecil.

4. Menghitung score hasil rekomendasi dengan *Intra Similarity*. Untuk penjelasan *Intra Similarity* akan dijelaskan pada tahapan evaluasi.

5. Membungkus *code*  proses 1 sampai 4 menjadi sebuah *class* agar lebih mudah untuk digunakan kembali. berikut *code* yang telah dibungkus menjadi sebuah *class*.

   ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\class_recsys.png)

   Kelebihan dari metode ini adalah mudah dalam implementasi sedangkan kekurangannya terkadang kurang akurat.

Pada proyek ini pembuatan model dilakukan dengan dua strategi yakni dengan top lima rekomendasi dengan pendekatan encoding *bag of word* dan *tfidf*.

- Model dengan *bag of word*

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling1.png)

  Karena *code* sudah dibungkus menjadi sebuah *class* maka implementasinya cukup mudah seperti gambar diatas. Setelah model dilatih tiba saatnya untuk melakukan *sanity check*. Data yang ditampilkan top 5 rekomendasi dari index ke 100 dan 1000.

  1. Data ke-100

     ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling1_1.png)

     Hasil rekomendasi dari data ke-100 ini memiliki score 43 persen.

  2. Data ke-1000

     ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling1_2.png)

     Hasil rekomendasi dari data ke-1000 memiliki score 73 persen

- Model dengan Tf-idf

  ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling2.png)

  Sama seperti pada *bag of word* data yang ditampilkan adalah data ke-100 dan data ke-1000.

  1. Data ke-100

     ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling2_1.png)

     Hasil rekomendasi dari data ke-100 ini memiliki score 49 persen.

  2. Data ke-1000

     ![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\modeling2_2.png)

     Hasil rekomendasi dari data ke-100 ini memiliki score 83 persen.
     
     Jika dilihat dari dua contoh sampel data terlihat model dengan *encoding* *tf-idf* lebih baik. Namun bagaimana jika sampel diperbanyak ? Pada tahap ini belum bisa disimpulkan model mana yang akan diajukan.

  

## Evaluation

Seperti yang sudah dipaparkan pada tahapan *solution approach* *metrics* yang akan digunakan dalam mengukur performa dari model pada proyek ini adalah dengan *intra similarity*. Berikut tahapan untuk lebih memahami *metric* *intra similarity*.

1. Melakukan rekomendasi top-N pada dokumen ke-K (dengan K bilangan bulat dari nol hingga jumlah data pada dataset).
2. Mengukur nilai cosine similarity antara dokumen ke-K dengan hasil rekomendasi top-N, kemudian rata-ratakan.
3. Ulangi tahap satu dan dua pada data berikutnya hingga jumlah sampel terpenuhi. Misalkan sampel 100 data maka ulangi tahap satu dan dua dengan data berbeda yang dipilih secara acak sebanyak 100 kali.
4. Setelah rata-rata telah didapat kemudian rata-ratakan kembali dan hasilnya merupakan score akhir.

Pada proyek ini jumlah data buku sebanyak kurang lebih 270 ribu data, akan sangat lama jika menggunakan seluruh dara untuk menghitung score dengan *intra similarity*. oleh karenanya digunakan lah beberapa sampel saja. Agar adil sampel data yang digunakan untuk mengukur performa kedua model akan sama yakni sebanyak 500 data. 

![](D:\Pemrograman\Python\Dicoding\Submission_MLT2\image\score_akhir.png)

Setelah dihitung score dengan 500 data berbeda sebanyak 2 kali model dengan *encoding bag of word* selalu mengungguli model dengan *encoding tf-idf* walau tidak signifikan. Maka pada proyek *recommendation system* buku berdasarkan kemiripan atribut ini model yang dipilih adalah model dengan *encoding bag of word*.


## Referensi
1. [Dicoding](https://www.dicoding.com/academies/319) (2021). *Machine learning Terapan*
2. [Scikit-learn](https://scikit-learn.org/). *Documentation*
3. Setyawatira, Rina. 2009. “Kondisi Minat Baca Di Indonesia.” *Jurnal Media Pustakawan* 16(1&2): 28–33. https://ejournal.perpusnas.go.id/mp/article/view/904/882.
4. Witanto, Janan. 2018. “Rendahnya Minat Baca Mata Kuliah Manajemen Kurikulum.” *Jurnal Perpustakaan Librarian* (April). https://www.researchgate.net/publication/324182095_Rendahnya_Minat_Baca.

5. Philip, Simon, P.B. Shola, and Abari Ovye. 2014. “Application of Content-Based Approach in Research Paper Recommendation System for a Digital Library.” *International Journal of Advanced Computer Science and Applications* 5(10): 37–40.

