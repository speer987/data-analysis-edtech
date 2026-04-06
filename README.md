```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Notebook Information and Motivation

### Table of Contents
1) Introduction to Exploration
2) Objectives & Questions
3) Data Loading
4) Dataset Information
5) Preliminary Explorations with Pandas

### Objectives & Questions

### Data Loading


```python
df = pd.read_csv('online_learning_engagement_dataset.csv')
```

### Dataset Information


I found a relevant dataset to use for this exploration on Kaggle here: 
https://www.kaggle.com/datasets/ssssws/online-learning-engagement-dataset

We can also see the number of rows and columns, and the column or attribute names themselves  of the dataset here:

##### Checking to see the rows and column count of the CSV file.


```python
print("Row and Column Count:", df.shape)
print("Attribute/Column Names:", df.columns)
```

    Row and Column Count: (50000, 18)
    Attribute/Column Names: Index(['student_id', 'age', 'gender', 'country', 'device_type',
           'internet_speed_mbps', 'study_hours_weekly', 'login_frequency_weekly',
           'avg_session_duration_min', 'video_watch_time_min',
           'assignments_submitted', 'forum_posts', 'quiz_attempts',
           'avg_quiz_score', 'attendance_rate', 'engagement_score', 'final_grade',
           'dropout'],
          dtype='object')


Based on the above output, we can see that there are 50000 rows, and 18 columns, which lines up with the CSV details on Kaggle. Additionally, the 18 column names (or attributes as sometimes called in Machine Learning) can be seen under the shape output.

### Data Cleaning

### Exploratory Data Analysis (EDA)
Below is some python code I wrote as a preliminary exploration of the data using pandas.

First, I'm interested in seeing if the table loaded, and what the columns look like, so I'll display the first five rows of the dataset to see some sample rows.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>student_id</th>
      <th>age</th>
      <th>gender</th>
      <th>country</th>
      <th>device_type</th>
      <th>internet_speed_mbps</th>
      <th>study_hours_weekly</th>
      <th>login_frequency_weekly</th>
      <th>avg_session_duration_min</th>
      <th>video_watch_time_min</th>
      <th>assignments_submitted</th>
      <th>forum_posts</th>
      <th>quiz_attempts</th>
      <th>avg_quiz_score</th>
      <th>attendance_rate</th>
      <th>engagement_score</th>
      <th>final_grade</th>
      <th>dropout</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>24</td>
      <td>Female</td>
      <td>USA</td>
      <td>Laptop</td>
      <td>44.70</td>
      <td>15.92</td>
      <td>10</td>
      <td>40.052752</td>
      <td>294.099759</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>46.69</td>
      <td>0.93</td>
      <td>8.046499</td>
      <td>22.447641</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>37</td>
      <td>Female</td>
      <td>Germany</td>
      <td>Tablet</td>
      <td>61.10</td>
      <td>6.37</td>
      <td>8</td>
      <td>32.442671</td>
      <td>400.397658</td>
      <td>7</td>
      <td>14</td>
      <td>5</td>
      <td>62.65</td>
      <td>0.59</td>
      <td>6.312988</td>
      <td>39.749905</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>46</td>
      <td>Female</td>
      <td>Australia</td>
      <td>Tablet</td>
      <td>43.10</td>
      <td>6.64</td>
      <td>7</td>
      <td>43.614509</td>
      <td>190.239738</td>
      <td>1</td>
      <td>14</td>
      <td>5</td>
      <td>58.42</td>
      <td>0.43</td>
      <td>4.143199</td>
      <td>31.061688</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>32</td>
      <td>Male</td>
      <td>India</td>
      <td>Tablet</td>
      <td>26.99</td>
      <td>10.91</td>
      <td>5</td>
      <td>30.697263</td>
      <td>370.451629</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>61.21</td>
      <td>0.65</td>
      <td>6.125258</td>
      <td>41.300634</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>28</td>
      <td>Male</td>
      <td>India</td>
      <td>Laptop</td>
      <td>52.28</td>
      <td>7.41</td>
      <td>6</td>
      <td>47.402999</td>
      <td>151.341296</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>74.92</td>
      <td>0.55</td>
      <td>4.979706</td>
      <td>39.148998</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



##### NOTES FOLLOWING THIS (REMOVE LATER):
Start with context and objective → shows business understanding
Show data cleaning & analysis → demonstrates practical skills
Visualize insights clearly → makes it engaging and readable
Conclude with insights & next steps → shows analytical thinking

##### LAYOUT FOR THIS PORTFOLIO
Title & Introduction
Objective / Questions
Data Loading
Data Cleaning / Preparation
Exploratory Data Analysis (EDA)
Visual Analysis (matplotlib + seaborn)
Key Insights
Conclusion / Next Steps

Optional: Dataset link, interactive plots, or extra analysis if relevant

### Notebook title (project name)
Short description of the project
Dataset source and context
Optional: an image or logo for visual appeal

### Objective / Questions
What you are trying to find out
Specific questions your analysis will answer

<h3 id="data-loading" style="color: mediumpurple;"> Data Loading </h3>
Load the dataset with pandas
Show basic info (head(), info(), describe())

### Data Cleaning / Preparation
Handle missing data
Filter rows/columns
Create new columns if needed
Group or aggregate data

### Exploratory Data Analysis (EDA)
Explore patterns in the data
Use pandas to summarize (groupby, value_counts)
Include matplotlib/seaborn plots for trends

### Visual Analysis
Use seaborn for cleaner, more advanced plots
Correlation heatmaps, scatterplots, boxplots, pairplots

### Key Insights
Summarize findings from analysis and plots
Use bullet points or markdown for clarity

### Conclusion / Next Steps
Wrap up project
Optional: suggest further analysis or improvements
