# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
## Data_set.csv
```
import pandas as pd
import numpy as np

df = pd.read_csv("Data_set.csv")

print("\nOriginal Data:\n", df.head(10))
print("\nMissing values:\n", df.isnull().sum())

clean = df.copy()

for col in clean.select_dtypes(include="number").columns:
    clean[col] = clean[col].fillna(clean[col].median())

for col in clean.select_dtypes(include="object").columns:
    if not clean[col].mode().empty:
        clean[col] = clean[col].fillna(clean[col].mode()[0])

print("\nMissing values after cleaning:\n", clean.isnull().sum())
print("\nCleaned Data:\n", clean.head(10))

clean.to_csv("Data_set_cleaned.csv", index=False)
print("\nSaved Data_set_cleaned.csv")
```
<img width="695" height="708" alt="Screenshot 2026-11-02 151143" src="https://github.com/user-attachments/assets/0ca08a33-0fd4-4afb-81de-0b39b9f624e4" />
<img width="383" height="450" alt="Screenshot 2026-11-02 151201" src="https://github.com/user-attachments/assets/924d7bc4-49ef-467f-8265-280a90d42767" />
<img width="785" height="747" alt="Screenshot 2026-11-02 151215" src="https://github.com/user-attachments/assets/d2ede9e0-43fb-4826-84bd-af766474f815" />

## SAMPLEIDS.csv
```

import pandas as pd
import numpy as np

df = pd.read_csv("SAMPLEIDS.csv")

print("\nOriginal Data:\n", df.head(10))
print("\nDuplicate rows:", df.duplicated().sum())

clean = df.drop_duplicates().copy()
for col in clean.select_dtypes(include="object").columns:
    clean[col] = clean[col].str.strip()
    print("\nUnique values in", col)
    print(clean[col].unique())

print("\nDuplicate rows after removal:", clean.duplicated().sum())
print("\nCleaned Data:\n", clean.head(10))

clean.to_csv("SAMPLEIDS_cleaned.csv", index=False)
print("\nSaved SAMPLEIDS_cleaned.csv")

```
<img width="803" height="482" alt="Screenshot 2026-11-02 151711" src="https://github.com/user-attachments/assets/3ef640b7-7507-4dcf-82d2-925f1def59e4" />
<img width="760" height="377" alt="Screenshot 2026-11-02 151722" src="https://github.com/user-attachments/assets/7b8047e1-dbec-42e5-aded-ca61892a6c9b" />
<img width="875" height="523" alt="Screenshot 2026-11-02 151731" src="https://github.com/user-attachments/assets/176db271-bde9-4677-a2da-6c3e7a45116c" />

## heights.csv
```
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

print("\nOriginal Data:\n", df.head(10))

for col in df.select_dtypes(include="number").columns:

    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = values[(values < lower) | (values > upper)]

    print("\nIQR results for", col)
    print("Lower limit:", lower)
    print("Upper limit:", upper)
    print("Unusual measurements:\n", outliers)

for col in df.select_dtypes(include="number").columns:
    plt.figure()
    plt.boxplot(df[col].dropna())
    plt.title("Unusual measurements in " + col)
    plt.ylabel(col)
    plt.show()
```
<img width="450" height="398" alt="Screenshot 2026-11-02 224553" src="https://github.com/user-attachments/assets/661e0cef-349c-4883-90bf-856e51086b1d" />
<img width="782" height="506" alt="Screenshot 2026-11-02 224611" src="https://github.com/user-attachments/assets/3af223e4-9cfb-4188-8c14-04a5d7014f1b" />

## iris.csv
```

import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("iris.csv")

print("\nOriginal Data:\n", df.head(10))


for col in df.select_dtypes(include="number").columns:

    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = values[(values < lower) | (values > upper)]

    print("\nIQR outliers in", col)
    print(outliers if not outliers.empty else "None")

# STEP 3: Z-SCORE OUTLIER DETECTION
for col in df.select_dtypes(include="number").columns:

    values = df[col].dropna()

    if len(values) < 2:
        continue

    z = np.abs(stats.zscore(values))
    outliers = values[z > 3]

    print("\nZ-score outliers in", col)
    print(outliers if not outliers.empty else "None")

```
<img width="647" height="725" alt="Screenshot 2026-11-02 225006" src="https://github.com/user-attachments/assets/31512e61-47d1-49e6-89a5-70b2f7bdadc4" />

## loan_data.csv
```

import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("Loan_data.csv")

print("\nOriginal Data:\n", df.head(10))


for col in df.select_dtypes(include="number").columns:

    values = df[col].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = values[(values < lower) | (values > upper)]

    print("\nIQR outliers in", col)
    print(outliers if not outliers.empty else "None")

# STEP 3: Z-SCORE OUTLIER DETECTION
for col in df.select_dtypes(include="number").columns:

    values = df[col].dropna()

    if len(values) < 2:
        continue

    z = np.abs(stats.zscore(values))
    outliers = values[z > 3]

    print("\nZ-score outliers in", col)
    print(outliers if not outliers.empty else "None")

```


# Result
          <<include your Result here>>
