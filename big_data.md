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

Three libraries were used in this assignment to compare traditional and scalable approaches to big data handling :


| Library | Purpose |
|--------|--------|
| **Pandas** | Baseline (single-threaded).It is widely adopted and easy to use, but operates in a single-threaded environment, limiting its performance on large datasets.  |
| **Dask** | Scalable parallel processing  that processes data in partitions. It is designed to handle datasets that exceed memory capacity and support distributed computing. |
| **Polars** | High-performance multi-threading.It offers fast execution using a Rust-based engine |


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
### Discussion
The dataset consists of structured transaction data with no missing values. It includes multiple data types, making it suitable for applying different optimisation strategies.

## 4. Big Data Handling Strategies

In this notebook, five effective strategies are applied to handle large datasets.  
Part 1 focuses on implementing optimisation techniques using **Pandas**, while Part 2 compares the performance of three libraries: **Pandas, Dask, and Polars**.

**Part 1** uses a subset of **1 million records** to reduce memory usage and speed up development. This allows efficient testing and demonstration of each strategy without overloading the system.

**Part 2** uses the **full dataset** for each library to ensure a fair and realistic performance comparison. Evaluating all libraries under the same conditions provides accurate measurements of execution time and memory usage, and better reflects real-world data processing scenarios.

---

### 🔹 Part 1: Big Data Handling Strategies
- Load Less Data  
- Use Chunking  
- Optimise Data Types  
- Sampling  
- Parallel Processing with Scalable Libraries  


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
### Discussion
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
Sampling is used to reduce the size of a dataset by selecting a representative subset. This allows faster processing and analysis while maintaining the overall structure and characteristics of the data. It is commonly used during development and testing to avoid long execution times on large datasets.

```python
import pandas as pd
import time

start_time = time.time()

# Load dataset
df_full = pd.read_csv("synthetic_fraud_data.csv", nrows=1000000)

# Apply sampling (10%)
df_sample = df_full.sample(frac=0.1, random_state=42)

sampling_time = time.time() - start_time

# Memory usage
mem_sampling = df_sample.memory_usage(deep=True).sum() / (1024**2)

print("Original rows:", df_full.shape[0])
print("Sampled rows:", df_sample.shape[0])
print("Memory usage (sample):", round(mem_sampling, 2), "MiB")
print("Execution time:", round(sampling_time, 2), "seconds")

df_sample.head()
```

### Output
```python
Original rows: 1000000
Sampled rows: 100000
Memory usage: 109.74 MiB
Execution time: 11.27 seconds
```

### Discussion
Sampling significantly reduces the dataset size, resulting in faster execution time and lower memory usage compared to processing the full dataset. This makes it highly useful during the development and testing phase, where quick iterations are required.

However, sampling may not capture all patterns present in the full dataset, especially for rare events such as fraud cases. Therefore, while it improves efficiency, it should be used carefully and complemented with full dataset analysis for final results.
________________________________________

### 4.5 Parallel Processing with Scalable Libraries
Parallel processing improves performance by executing multiple operations simultaneously using multiple CPU cores instead of sequential processing. This is especially important for large datasets where single-threaded processing becomes slow and inefficient.


**Library Used**  
Polars is used as it supports multi-threaded execution by default. Unlike Pandas, which is single-threaded, Polars distributes operations across multiple CPU cores, resulting in faster and more efficient data processing.

```python
import polars as pl
import time

print("\n--- Parallel Processing using Polars ---")

start_time = time.time()

# Load dataset (multi-threaded internally)
df = pl.read_csv("synthetic_fraud_data.csv", n_rows=1_000_000)

# Measure memory usage
mem_before = df.estimated_size("mb")

# Parallel operations
result = (
    df
    .filter(pl.col("amount") > 100)
    .group_by("merchant_category")
    .agg([
        pl.col("amount").mean().alias("avg_amount"),
        pl.col("amount").count().alias("transaction_count")
    ])
)

end_time = time.time()

print(result.head())
print("\nExecution Time:", round(end_time - start_time, 2), "seconds")
print("Memory Usage (Before):", round(mem_before, 2), "MiB")
```

### Output
```python
--- Parallel Processing using Polars ---
shape: (5, 3)
┌───────────────────┬──────────────┬───────────────────┐
│ merchant_category │ avg_amount   │ transaction_count │
├───────────────────┼──────────────┼───────────────────┤
│ Education         │ 48041.03     │ 118790            │
│ Retail            │ 65623.42     │ 119067            │
│ Restaurant        │ 30708.82     │ 107907            │
│ Entertainment     │ 32767.58     │ 111443            │
│ Gas               │ 49028.25     │ 119424            │
└───────────────────┴──────────────┴───────────────────┘

Execution Time: 12.08 seconds
Memory Usage (Before): 338.77 MiB
```

### Discussion
Polars efficiently processes large datasets using parallel processing. By leveraging multi-threading, operations such as filtering, grouping, and aggregation are executed simultaneously across multiple CPU cores, resulting in faster execution compared to sequential processing.

The memory usage (~338 MiB) remains reasonable for 1 million records, showing that Polars handles data efficiently on a single machine.

However, Polars is mainly designed for single-machine processing. For much larger datasets that exceed memory limits, distributed frameworks such as Dask or Apache Spark may be more suitable.

Overall, this strategy demonstrates how scalable libraries improve performance and efficiency when working with big data.
________________________________________
### 🔹 Part 2: Loading Dataset with Different Libraries
This approach loads the entire dataset into memory using Pandas without applying any optimisation techniques. It represents the traditional method of handling data and serves as a baseline for comparison with other big data handling strategies.

- Pandas  
- Dask  
- Polars

### 1. 📦 Full Load Using Pandas (Traditional Approach)

```python
import pandas as pd
import time

start = time.time()

# Load complete dataset
df_pandas = pd.read_csv("synthetic_fraud_data.csv")

end = time.time()

# Calculate memory usage
mem_pandas = df_pandas.memory_usage(deep=True).sum() / (1024**2)

# Output
print(df_pandas.head(5))
print("Shape:", df_pandas.shape)
print(f"Execution Time: {end - start:.2f} seconds")
print(f"Memory usage: {mem_pandas:.2f} MiB")

```

### Output
```python
Shape: (7483766, 24)
Execution Time: 83.33 seconds
Memory usage: 8148.61 MiB
```

### Discussion
The full load approach using Pandas results in high memory consumption (~8148 MiB) and long execution time (~83 seconds). This is because the entire dataset is loaded into memory at once and processed using a single-threaded approach.

While this method is simple and easy to implement, it is not suitable for large datasets as it can quickly exhaust available memory and significantly slow down processing. This highlights the limitations of traditional data processing methods and the need for optimisation strategies such as chunking, sampling, and parallel processing.
________________________________________
### 2. 📦 Full Load Using Dask
This approach uses Dask to load and process the full dataset. Dask supports lazy evaluation, meaning data is not immediately loaded into memory. Instead, operations are deferred until explicitly executed. In this case, the `.compute()` function is used to force the dataset to be fully loaded into memory for comparison purposes.
```python
import dask.dataframe as dd
import time

start = time.time()

# Load dataset lazily
ddf = dd.read_csv("synthetic_fraud_data.csv")

# Force full load into memory
df_dask = ddf.compute()

end = time.time()

# Calculate memory usage
mem_dask = df_dask.memory_usage(deep=True).sum() / (1024**2)

# Output
print(df_dask.head(5))
print("Shape:", df_dask.shape)
print(f"Execution Time: {end - start:.2f} seconds")
print(f"Memory usage: {mem_dask:.2f} MiB")
```
### Output
```python
Shape: (7483766, 24)
Execution Time: 96.43 seconds
Memory usage: 3523.78 MiB
```

### Discussion
The Dask implementation shows lower memory usage (~3524 MiB) compared to Pandas, as it processes data in partitions rather than loading everything at once initially. However, the execution time (~96 seconds) is slower due to overhead from task scheduling and coordination.

When .compute() is called, the full dataset is still brought into memory, which reduces some of the advantages of Dask's lazy evaluation in this scenario. This explains why performance is slower compared to Pandas despite improved memory efficiency.

Overall, Dask is more suitable for handling larger-than-memory datasets and distributed computing environments. However, in a single-machine setup with limited resources, its overhead can result in slower performance.
________________________________________
### 3. 📦 Full Load Using Polars
This approach uses Polars to load the full dataset into memory. Polars is designed for high-performance data processing and uses a Rust-based engine with built-in multi-threading. This allows it to process data faster and more efficiently compared to traditional single-threaded libraries.

```python
import polars as pl
import time

# Measure execution time
start_time = time.time()

# Load complete dataset
df_polars = pl.read_csv("synthetic_fraud_data.csv")

# Execution time
exec_time = time.time() - start_time

# Measure memory usage
mem_used = df_polars.estimated_size() / (1024**2)

# Output
print("First 5 rows:", df_polars.head(5))
print("Shape:", df_polars.shape)
print("Execution Time:", round(exec_time, 2), "seconds")
print("Memory usage:", round(mem_used, 2), "MiB")
```

### Output 
```python
Shape: (7483766, 24)
Execution Time: 16.17 seconds
Memory usage: 2528.16 MiB
```

### Discussion
Polars demonstrates significantly faster performance (~16 seconds) compared to Pandas and Dask due to its multi-threaded execution and efficient Rust-based engine. It is able to process large datasets quickly by utilising multiple CPU cores.

The memory usage (~2528 MiB) is also lower than Pandas, indicating more efficient memory management. Compared to Dask, Polars achieves better speed while still maintaining relatively low memory usage.

Overall, Polars is highly suitable for high-performance data processing on a single machine. However, like Pandas, it operates within a single-node environment, and for extremely large datasets, distributed frameworks such as Dask or Apache Spark may be more appropriate.
________________________________________

### **📊 Overall Comparison**

The full dataset results show clear performance differences among the three libraries.

- 🐼 **Pandas** → Highest memory usage and slower execution due to single-threaded processing  
- ⚙️ **Dask** → Lower memory usage but slower execution due to scheduling overhead  
- ⚡ **Polars** → Fastest execution and lowest memory usage with multi-threaded processing  

**💡 Insight:**  
Polars is the most efficient for single-machine processing, while Dask is better suited for scalable, distributed environments.
________________________________________
## 5. Comparative Analysis
🔍 **Part 1: Comparison between 5 Big Data Handling Strategies**
- Load Less Data
- Use Chunking
- Optimize Data Types
- Sampling
- Parallel Processing with Polars

Two bar charts are generated:
-One compares the execution time (in seconds).
-The other compares the memory usage (in MB).

This analysis helps to identify which strategy offers the best trade-off between speed and memory efficiency when using traditional vs. parallelized approaches.

### Performance Table
🔍 **Part 2: Compare between 3 library**
In this section, the performance of three data processing libraries is evaluated:
- Pandas
- Polars
- Dask
This analysis provides insight into the trade-offs between performance, memory efficiency, and scalability across different libraries. The results are presented using tables and visualisations to clearly highlight performance differences.

### ⚡ Library Performance Comparison

| Library | Execution Time (s) | Memory Usage (MiB) |
|--------|------------------|-------------------|
| Pandas | 1.96 | 217.97 |
| Dask   | 2.77 | 92.85  |
| Polars | 0.39 | 64.62  |

---

### 📈 Visualisation


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







