# 📘 Assignment 2: Mastering Big Data Handling  

---

## 👥 Group Information  

**Group CC**  
- Chua Jia Lin (A23CS0069)  
- Joanne Ching Yin Xuan (A23CS0227)  

---

## 1. Dataset Description  

The dataset used in this assignment is the **Transactions Dataset** obtained from Kaggle.

- **Source**: https://www.kaggle.com/datasets/ismetsemedov/transactions/data  
- **File Size**: 2.73 GB  
- **Domain**: Finance 
- **Number of Records**: 7,483,766 rows  
- **Number of Columns**: 24

This dataset contains detailed transaction-level information such as customer data, merchant details, transaction amounts, and fraud indicators. The large size of the dataset makes it suitable for evaluating big data handling strategies and performance optimization techniques. 
This dataset was selected because it provides: 
- A sufficiently large volume of data to simulate real-world big data challenges   
- Diverse data types suitable for optimization techniques   
- A practical use case in financial analytics and fraud detection   
 
Therefore, it is well-suited for comparing the performance of Pandas, Dask, and Polars under large-scale data processing conditions. 

---

## 2. Library Choices  

Three libraries were used in this assignment:

- **Pandas**  
 Pandas is used as the baseline library. It is widely adopted and easy to use, but operates in a single-threaded environment, limiting its performance on large datasets. 

- **Dask**  
Dask is a scalable parallel computing library that processes data in partitions. It is designed to handle datasets that exceed memory capacity and support distributed computing. 

- **Polars**  
Polars is a high-performance DataFrame library built in Rust. It supports multi-threading and efficient memory usage through columnar storage, making it highly suitable for large-scale data processing.

These libraries were selected to compare traditional and scalable approaches to big data handling.  

---

## 3. Data Loading and Inspection  

### 3.1 Loading Dataset  

```python
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ismetsemedov/transactions
!unzip transactions.zip
```


### 3.2 Load Sample (1 Million Rows)
```python
import pandas as pd
df = pd.read_csv("synthetic_fraud_data.csv", nrows=1000000)
```


### 3.3 Inspect Dataset
```python
print("First 5 rows:")
display(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

```
Output
```python
Shape: (1000000, 24)
No missing values detected
Data types include: object, float64, int64, bool

```
Discussion
The dataset consists of structured transaction data with no missing values. It includes multiple data types, making it suitable for applying different optimisation strategies.

## 4. Big Data Handling Strategies
### 4.1 Load Less Data
```python
selected_cols = [
    'transaction_id','customer_id','merchant',
    'amount','currency','high_risk_merchant','is_fraud'
]

df_selected = pd.read_csv(
    "synthetic_fraud_data.csv",
    usecols=selected_cols,
    nrows=1000000
)
```
### Output
```python
Shape: (1000000, 7)
Memory usage: ~228 MB
Execution time: ~4.13 seconds
```
### Explanation
Loading only necessary columns reduces memory usage and improves performance.

### 4.2 Chunking
```python
chunk_iter = pd.read_csv(
    "synthetic_fraud_data.csv",
    chunksize=200000,
    nrows=1000000
)
```

### Output
```python
Processed in chunks
Max memory usage: ~217 MB
Execution time: ~14.7 seconds
```

### Explanation
Chunking helps process large datasets without exceeding memory limits but adds processing overhead.
________________________________________

### 4.3 Data Type Optimisation
```python
df = pd.read_csv("synthetic_fraud_data.csv", nrows=1000000)

df["amount"] = df["amount"].astype("float32")
df["transaction_hour"] = df["transaction_hour"].astype("int8")
```
### Output
```python
Memory usage reduced
Execution time: ~17.7 seconds
```

### Discussion
Optimising data types significantly reduces memory usage but requires careful implementation.

________________________________________
### 4.4 Sampling
```python
df_sample = df.sample(frac=0.1, random_state=42)
```

### Output
```python
Original rows: 1000000
Sampled rows: 100000
Memory usage: ~109 MB
Execution time: ~10.4 seconds
```

Discussion
Sampling reduces computation time but may not fully represent the entire dataset.
________________________________________

### 4.5 Parallel Processing
```python
# Pandas
df_pandas = pd.read_csv(sample_file)

# Dask
ddf = dd.read_csv(sample_file)
df_dask = ddf.compute()

# Polars
df_polars = pl.read_csv(sample_file)
```

### Output
```python
Pandas: Time ~2.45s | Memory ~217.97 MiB
Dask:   Time ~3.64s | Memory ~92.85 MiB
Polars: Time ~0.52s | Memory ~64.62 MiB
```

### Explanation
Polars performs the fastest due to multi-threading. Dask provides scalability but has overhead. Pandas is simple but less efficient.

________________________________________
## 5. Comparative Analysis
### Performance Table
| Library | Execution Time (s) | Memory (MiB) |
|--------|------------------|-------------|
| Pandas | 2.45             | 217.97      |
| Dask   | 3.64             | 92.85       |
| Polars | 0.52             | 64.62       |
________________________________________

### Analysis
Polars achieved the best performance due to efficient memory usage and multi-threading. Pandas showed moderate performance but used the most memory. Dask introduced overhead, making it slower in this environment, but it remains scalable for large datasets.

________________________________________
## 6. Conclusion and Reflection
This assignment highlights the importance of selecting appropriate strategies for handling large datasets.
•	Load Less Data and Sampling improved speed 
•	Data Type Optimisation reduced memory 
•	Parallel Processing improved scalability 

Polars provided the best overall performance, while Dask is suitable for large-scale distributed processing.
Reflection

The key learning is that performance depends on both the dataset size and system limitations. Tools like Dask may not perform well in smaller environments but are powerful for large-scale processing.
Scalability

For very large datasets:
•	Pandas is not suitable 
•	Polars works well on single machines 
•	Dask is best for distributed systems 
________________________________________
References

•	Kaggle Transactions Dataset 

•	Pandas Documentation 

•	Dask Documentation 

•	Polars Documentation 

---







