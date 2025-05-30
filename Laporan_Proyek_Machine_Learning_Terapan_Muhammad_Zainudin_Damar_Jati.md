# Laporan Proyek Machine Learning - Muhammad Zainudin Damar Jati

## Domain Proyek

Prediksi harga rumah menjadi hal penting dalam pengambilan keputusan di bidang properti, investasi, dan perencanaan keuangan. Dengan semakin kompleksnya faktor yang memengaruhi harga rumah, teknologi machine learning dapat memberikan solusi berbasis data yang lebih akurat dan efisien. Proyek ini bertujuan membangun model prediktif menggunakan data perumahan di California dengan fitur-fitur demografis dan karakteristik properti.

Referensi:
Zhang, Y., Guhathakurta, S., & Khalil, E.B. (2019). A machine learning approach to real estate price prediction: A case study of Atlanta housing market. *Journal of Real Estate Research*.

## Business Understanding

### Problem Statements

1. Bagaimana membangun model prediksi harga rumah yang akurat dari data demografi dan properti?
2. Seberapa baik performa model dalam memprediksi harga rumah dibandingkan data sebenarnya?
3. Fitur-fitur mana yang paling berkontribusi terhadap harga rumah di California berdasarkan analisis Permutation Importance?

### Goals

1. Mengembangkan model machine learning untuk memprediksi harga rumah secara akurat.
2. Menilai performa model menggunakan metrik R², RMSE, MAE, dan validasi silang.
3. Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap harga rumah menggunakan analisis Permutation Importance dari model.

### Solution Statements

* Menggunakan algoritma HistGradientBoostingRegressor sebagai model utama.
* Melakukan preprocessing data: interpolasi nilai hilang, feature engineering, penanganan outlier.
* Menerapkan hyperparameter tuning untuk mendapatkan performa terbaik dari model.

## Data Understanding

Dataset digunakan dari [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Dataset ini berisi data rumah-rumah di California berdasarkan sensus tahun 1990 dengan 20.640 entri dan 10 fitur utama.

### Fitur:

* `longitude`, `latitude`: Lokasi geografis
* `housing_median_age`: Usia median rumah
* `total_rooms`, `total_bedrooms`: Jumlah ruangan dan kamar tidur
* `population`, `households`: Jumlah penduduk dan rumah tangga
* `median_income`: Pendapatan median
* `ocean_proximity`: Kedekatan rumah terhadap laut (kategori)


### Target:

* `median_house_value` : Harga rumah median

### Feature Engineering:

* `rooms_per_household` : Rasio ruangan terhadap rumah tangga
* `bedrooms_per_room` : Rasio kamar tidur terhadap ruangan
* `population_per_household` : Rasio populasi terhadap rumah tangga

### Kondisi Missing Value:

* Fitur `total_bedrooms` memiliki 207 nilai hilang dari total 20.640 entri. Informasi ini penting dalam merancang strategi penanganan missing value pada tahap berikutnya.

### Visualisasi Awal:

#### Distribusi Fitur Numerik

![Distribusi Fitur Numerik](imeges/distribusi%20fitur%20numerik.png)

Grafik ini menggambarkan distribusi dari 12 fitur numerik dalam dataset California Housing. Sebagian besar fitur menunjukkan pola distribusi yang tidak normal (miring ke kanan), terutama pada variabel absolut seperti `total_rooms`, `population`, dan `households`, yang menunjukkan adanya outlier. Sementara itu, fitur rasio seperti `bedrooms_per_room` dan `population_per_household` cenderung memiliki distribusi yang lebih mendekati normal. Lonjakan di nilai tertentu seperti pada `housing_median_age` dan `median_house_value` mengindikasikan adanya batas data, seperti sensor atau clipping. Titik-titik mendekati garis diagonal menandakan akurasi yang baik, meskipun error meningkat pada harga tinggi karena heteroskedastisitas.


#### Korelasi antar Fitur Numerik

![Korelasi Fitur Numerik](imeges/korelasi%20antar%20fitur%20numerik.png)

Grafik ini adalah heatmap korelasi Pearson antar fitur numerik. Terlihat beberapa korelasi sangat tinggi seperti `total_rooms` dengan `total_bedrooms`, serta `households` dengan `population`. Hal ini menunjukkan potensi multikolinearitas dalam model regresi jika semua fitur digunakan tanpa seleksi. Korelasi negatif seperti `bedrooms_per_room` terhadap `median_income` memberikan insight bahwa daerah dengan rasio kamar tidur tinggi cenderung berpenghasilan lebih rendah. Korelasi lemah antara `median_house_value` dan fitur seperti `total_rooms` menunjukkan bahwa nilai rumah tidak hanya ditentukan oleh ukuran fisik, melainkan juga faktor eksternal seperti lokasi atau pendapatan.


## Data Preparation

Tahapan data preparation dilakukan sebagai berikut:

* Interpolasi untuk mengisi nilai hilang pada `total_bedrooms`.
* SimpleImputer dengan strategi median digunakan pada fitur numerik lainnya melalui pipeline.
* Feature Engineering untuk menambahkan fitur rasio: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`.
* Transformasi logaritmik (`np.log1p`) diterapkan pada target `median_house_value` untuk menstabilkan variansi.
* Penanganan Outlier dengan metode IQR untuk fitur numerik.
* One-Hot Encoding diterapkan pada fitur kategorik `ocean_proximity`.
* Scaling fitur numerik menggunakan MinMaxScaler (bukan StandardScaler seperti yang semula tertulis).
* Pembagian data dilakukan sebelum modeling: 80% data latih dan 20% data uji.

## Modeling

Model utama yang digunakan adalah HistGradientBoostingRegressor, yaitu algoritma ansambel berbasis boosting yang membangun decision tree secara bertahap. Model ini memanfaatkan histogram-based binning untuk efisiensi dan performa yang lebih baik, terutama pada dataset besar.

### Proses Modeling:

* Pemisahan data menjadi `X` (fitur) dan `y` (target)
* Preprocessing melalui Pipeline dan ColumnTransformer:

  * Numerik: Imputasi median, scaling
  * Kategorik: One-Hot Encoding
* Penerapan RandomizedSearchCV untuk tuning hyperparameter dengan parameter grid:

  * `max_iter`: \[100, 200, 300]
  * `max_leaf_nodes`: \[31, 50, 100]
  * `learning_rate`: \[0.05, 0.1, 0.2]

Nilai terbaik dari hasil tuning yang digunakan untuk model final:

* `max_iter = 300`
* `max_leaf_nodes = 100`
* `learning_rate = 0.05`

## Evaluation

### Hasil Evaluasi Model Final:

* R² Score: 0.8435 → Model menjelaskan 84.35% variansi target
* RMSE: \$44,481 → Deviasi prediksi dari nilai aktual rata-rata sebesar \$44.481
* MAE: \$28,332 → Rata-rata kesalahan absolut sebesar \$28.332

### Cross-Validation:

* R² Score rata-rata: 0.7014 ± 0.0602, menunjukkan stabilitas performa model terhadap data baru.

### Visualisasi Evaluasi:

#### Prediksi vs Nilai Aktual (Scatter Plot)

![Prediksi vs Realita](imeges/prediksivrealita.png)

Scatter plot ini menunjukkan hubungan antara harga rumah aktual dan hasil prediksi model. Sebagian besar titik berada di dekat garis ideal, yang menandakan prediksi cukup akurat. Namun, terlihat bahwa error cenderung meningkat pada harga rumah yang lebih tinggi, mengindikasikan heteroskedastisitas. Penumpukan titik di ujung atas grafik menunjukkan adanya batas maksimum harga rumah dalam dataset, yang dapat memengaruhi performa model pada segmen harga tinggi.


#### Distribusi Residual

![Distribusi Residual](imeges/distribusi%20residual.png)

Grafik histogram ini memperlihatkan distribusi residual model regresi. Distribusinya tampak simetris dan berbentuk lonceng, dengan rata-rata mendekati nol, menunjukkan bahwa kesalahan model tersebar secara acak tanpa bias sistematis. Hal ini menunjukkan bahwa model cukup baik dalam memprediksi data secara umum dan residual mengikuti asumsi normalitas, yang penting untuk validitas beberapa metode statistik lanjutan.


#### Feature Importance

![Feature Importance](imeges/feature_importance.png)

**Visualisasi Feature Importance:**
Grafik di atas menunjukkan 15 fitur terpenting yang berkontribusi dalam prediksi harga rumah menurut model *HistGradientBoostingRegressor*. Beberapa insight utama dari grafik tersebut:

* `num__latitude` merupakan fitur paling penting, yang mengindikasikan bahwa **lokasi geografis rumah (utara–selatan)** sangat memengaruhi harga rumah di California.
* `num__median_income` berada di posisi kedua, menegaskan bahwa **pendapatan median wilayah** berperan besar dalam menentukan nilai properti.
* Fitur `num__longitude` dan `cat__ocean_proximity_INLAND` juga memiliki kontribusi signifikan, menunjukkan bahwa **lokasi barat–timur** dan **kedekatan terhadap laut (khususnya wilayah non-pesisir atau pedalaman)** turut memengaruhi harga.
* Fitur turunan dari proses feature engineering seperti `num__population_per_household` dan `num__bedrooms_per_room` termasuk dalam daftar fitur penting, menunjukkan bahwa rasio demografis dan kepadatan rumah tangga juga berdampak terhadap harga rumah.
* Beberapa fitur lainnya seperti `num__total_bedrooms`, `num__population`, dan `cat__ocean_proximity_NEAR OCEAN` memiliki kontribusi yang relatif kecil menurut hasil Permutation Importance.
## Deployment: Prediksi Rumah Baru

### Input Rumah Baru:

* `longitude`: -118.0
* `latitude`: 34.0
* `housing_median_age`: 30
* `total_rooms`: 2000
* `total_bedrooms`: 400
* `population`: 800
* `households`: 300
* `median_income`: 5.0
* `ocean_proximity`: "NEAR OCEAN"

### Output Prediksi:

* Harga Rumah (prediksi): \$290,653.41

Model mampu memberikan estimasi harga rumah yang realistis berdasarkan fitur input, dan dapat digunakan dalam pengambilan keputusan sektor properti.

## Kesimpulan

Proyek ini berhasil membangun model prediksi harga rumah di California dengan akurasi yang tinggi menggunakan algoritma **HistGradientBoostingRegressor**. Model akhir mampu menjelaskan **lebih dari 84% variansi** harga rumah dan menghasilkan **kesalahan prediksi yang relatif kecil**, seperti ditunjukkan oleh nilai RMSE dan MAE.

Seluruh tahapan — mulai dari eksplorasi data, penanganan missing value, feature engineering, transformasi target, hingga tuning hyperparameter — dilakukan secara sistematis dan terstruktur. Hasil evaluasi model melalui validasi silang juga menunjukkan **performa yang stabil** terhadap data baru.

Lebih lanjut, analisis **Permutation Importance** mengungkapkan bahwa fitur `latitude` merupakan **kontributor paling dominan** dalam prediksi harga rumah, diikuti oleh `median_income`, `longitude`, dan `ocean_proximity_INLAND`. Hal ini menunjukkan bahwa **lokasi geografis dan kondisi ekonomi wilayah** merupakan faktor penentu utama dalam penilaian properti di California. Selain itu, fitur turunan seperti `population_per_household` dan `bedrooms_per_room` juga terbukti relevan, menegaskan pentingnya rasio demografis dalam model prediksi.

Dengan performa yang baik dan interpretabilitas yang jelas, model ini **siap di-deploy** sebagai alat bantu pengambilan keputusan di sektor properti dan investasi.


**Dataset**: [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

