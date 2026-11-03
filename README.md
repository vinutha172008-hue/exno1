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
<img width="678" height="235" alt="Screenshot 2026-11-03 113515" src="https://github.com/user-attachments/assets/a69eb4c3-2b99-4d6b-9675-feca4c8b37d0" />


<img width="700" height="205" alt="Screenshot 2026-11-03 113524" src="https://github.com/user-attachments/assets/cbc8e2b6-a803-4ff8-bf19-f0ee6e443729" />


<img width="410" height="326" alt="Screenshot 2026-11-03 113536" src="https://github.com/user-attachments/assets/82f8d1fb-29cf-49c0-82b9-aa9febbf5d86" />


<img width="388" height="327" alt="Screenshot 2026-11-03 113551" src="https://github.com/user-attachments/assets/1a2ab532-2b98-4aae-8e20-8a096b4b5760" />


<img width="382" height="332" alt="Screenshot 2026-11-03 113602" src="https://github.com/user-attachments/assets/cad087b9-4952-46a7-af8e-a1dec82c8cf0" />


<img width="347" height="331" alt="Screenshot 2026-11-03 113614" src="https://github.com/user-attachments/assets/e4e0fcf2-03f8-4f30-909b-09295f62e60c" />


<img width="298" height="312" alt="Screenshot 2026-11-03 113624" src="https://github.com/user-attachments/assets/298a35e7-21fd-4cc6-b4b1-f4e085ae5c6b" />


<img width="390" height="313" alt="Screenshot 2026-11-03 113635" src="https://github.com/user-attachments/assets/6d583edf-91e9-49be-a6b6-8c3e3dbdeaa7" />


<img width="417" height="325" alt="Screenshot 2026-11-03 113648" src="https://github.com/user-attachments/assets/50ab9f27-c6ab-4ff2-b604-aa0233ae4323" />


<img width="442" height="322" alt="Screenshot 2026-11-03 113702" src="https://github.com/user-attachments/assets/c82895a1-60e7-4bdb-ab35-435850004fe4" />


<img width="390" height="340" alt="Screenshot 2026-11-03 113716" src="https://github.com/user-attachments/assets/28ab3cdd-fadd-4556-8c1c-546be7c8d009" />


# Result
   The code is successfully executed.
