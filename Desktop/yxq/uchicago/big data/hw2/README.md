ssh –Y sarah@hadoop.rcc.uchicago.edu

1-1
```
wget https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD
mv rows.csv crimes.csv
hdfs dfs -put crimes.csv data/crimes.csv
```

1-2
```sql
create database sarahyang;
use sarahyang;
DROP TABLE IF EXISTS chicago_crimes;
create external table chicago_crimes (id int, case_number string, m_date string, block string, iucr string, primary_type string, description string, location_description string, arrest boolean, domestic boolean, beat string, district string, ward int, community_area int, fbi_code int, x_coordinate float, y_coordinate float, year int, updated_on timestamp, latitude float, longitude float, location string) row format delimited fields terminated by ',' LINES TERMINATED BY '\n' STORED AS TEXTFILE tblproperties("skip.header.line.count"="1");
```

```sql
use sarahyang;
DROP TABLE IF EXISTS chicago_covid;
create external table chicago_covid (Dates string, Cases_Total int, Deaths_Total int, Hospitalizations_Total int, Cases_Age_0_17 int, Cases_Age_18_29 int, Cases_Age_30_39 int, Cases_Age_40_49 int, Cases_Age_50_59 int, Cases_Age_60_69 int, Cases_Age_70_79 int, Cases_Age_80 int, Cases_Age_Unknown int, Cases_Female int, Cases_Male int, Cases_Unknown_Gender int, Cases_Latinx int, Cases_Asian_Non_Latinx int, Cases_Black_Non_Latinx int, Cases_White_Non_Latinx int, Cases_Other_Race_Non_Latinx int, Case_Unknown_Race_Ethnicity int, Deaths_Age_0_17 int, Deaths_Age_18_29 int, Deaths_Age_30_39 int, Deaths_Age_40_49 int, Deaths_Age_50_59 int, Deaths_Age_60_69 int, Deaths_Age_70_79 int, Deaths_Age_80 int, Deaths_Age_Unknown int, Deaths_Female int, Deaths_Male int, Deaths_Unknown_Gender int, Deaths_Latinx int, Deaths_Asian_Non_Latinx int, Deaths_Black_Non_Latinx int, Deaths_White_Non_Latinx int, Deaths_Other_Race_Non_Latinx int, Deaths_Unknown_Race_Ethnicity int, Hospitalizations_Age_0_17 int, Hospitalizations_Age_18_29 int, Hospitalizations_Age_30_39 int, Hospitalizations_Age_40_49 int, Hospitalizations_Age_50_59 int, Hospitalizations_Age_60_69 int, Hospitalizations_Age_70_79 int, Hospitalizations_Age_80 int, Hospitalizations_Age_Unknown int, Hospitalizations_Female int, Hospitalizations_Male int, Hospitalizations_Unknown_Gender int, Hospitalizations_Latinx int, Hospitalizations_Asian_Non_Latinx int, Hospitalizations_Black_Non_Latinx int, Hospitalizations_White_Non_Latinx int, Hospitalizations_Other_Race_Non_Latinx int, Hospitalizations_Unknown_Race_Ethnicity int) row format delimited fields terminated by ',' LINES TERMINATED BY '\n' STORED AS TEXTFILE tblproperties("skip.header.line.count"="1");
```

note: date is a data type

1-3

```sql
LOAD DATA INPATH '/user/sarahyang/data/crimes.csv' INTO TABLE chicago_crimes;
```

2-4
```sql
select c.m_date, c.primary_type
from
(select m_date, primary_type, case when m_date is null then 0 else 1 end as flg
from chicago_crimes
order by flg desc, m_date asc) c
limit 1;
```
2001-01-01 00:00:00	THEFT
Note: watch null values

```sql
select c.m_date, c.primary_type
from chicago_crimes c
order by c.m_date desc
limit 1;
```
2021-04-18 23:55:00	NARCOTICS

2-5
```sql
SELECT c.primary_type, c.count FROM
    (SELECT primary_type, COUNT(*) AS count from chicago_crimes
    GROUP BY primary_type
    ) c
ORDER BY c.count
LIMIT 5;
```
Last 5

"iucr":"2017"	1
"iucr":"0880"	1
"iucr":"3731"	1
"iucr":"0330"	1
"iucr":"5007"	1

```sql
SELECT c.primary_type, c.count FROM
    (SELECT primary_type, COUNT(*) AS count from chicago_crimes
    GROUP BY primary_type
    ) c
ORDER BY c.count DESC
LIMIT 5;
```
Top 5

THEFT	15208177
BATTERY	13188092
CRIMINAL DAMAGE	8207601
NARCOTICS	7336907
ASSAULT	4549512

2-6
```sql
SELECT location_description, COUNT(*) AS count FROM chicago_crimes
WHERE primary_type = 'HOMICIDE'
GROUP BY location_description
ORDER BY count DESC
LIMIT 1;
```
STREET	51280

2-7

```sql
SELECT district, COUNT(*) AS m_count FROM chicago_crimes
WHERE district IS NOT NULL
GROUP BY district
ORDER BY m_count DESC
LIMIT 1;
```
Most dangerous: 008	4697293

```sql
SELECT district, COUNT(*) AS m_count FROM chicago_crimes
WHERE district IS NOT NULL
GROUP BY district
ORDER BY m_count
LIMIT 10;
```
Least dangerous: 022 and 020: 27

2-8
```sql
SELECT COUNT(*) / 12 AS monthly_assult FROM chicago_crimes
WHERE primary_type = 'ASSAULT' AND m_date BETWEEN '2019-01-01 00:00:00.00' AND '2020-01-01 00:00:00.00';
```
2019 monthly assault count: 1718.3333333333333

```sql
SELECT COUNT(*) / 12 AS monthly_assult FROM chicago_crimes
WHERE primary_type = 'ASSAULT' AND m_date BETWEEN '2018-01-01 00:00:00.00' AND '2019-01-01 00:00:00.00';
```
2018 monthly assault count: 1700.4166666666667

2-9

1)
<!-- ```sql
INSERT OVERWRITE DIRECTORY '/user/sarahyang/data/q10.csv' row format delimited fields terminated by ',' LINES TERMINATED BY '\n' STORED AS TEXTFILE
SELECT community_area, COUNT(*) FROM chicago_crimes
WHERE primary_type = 'OFFENSE INVOLVING CHILDREN'
GROUP BY community_area;
```
hadoop fs -get /user/sarahyang/data/q10.csv q10.csv -->
<!--
```SQL
create external table child (community_area int, count int) row format delimited fields terminated by ',' LINES TERMINATED BY '\n' STORED AS TEXTFILE tblproperties("skip.header.line.count"="1");

INSERT OVERWRITE TABLE child SELECT community_area, COUNT(*) FROM chicago_crimes
WHERE primary_type = 'OFFENSE INVOLVING CHILDREN'
GROUP BY community_area;
```
hive -e 'use sarahyang; select * from child' | sed 's/[\t]/,/g' >> q10.csv -->

2)
<!-- ```SQL
INSERT OVERWRITE DIRECTORY '/user/sarahyang/data/q11.csv' row format delimited fields terminated by ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE
SELECT primary_type, community_area, COUNT(*) FROM chicago_crimes
GROUP BY primary_type, community_area;
```
hadoop fs -get /user/sarahyang/data/q11.csv q11.csv -->


```sql
create external table area (primary_type string, community_area int, count int) row format delimited fields terminated by ',' LINES TERMINATED BY '\n' STORED AS TEXTFILE tblproperties("skip.header.line.count"="1");

INSERT OVERWRITE TABLE area SELECT primary_type, community_area, COUNT(*) FROM chicago_crimes
GROUP BY primary_type, community_area;
```
hive -e 'use sarahyang; select * from area' | sed 's/[\t]/,/g' >> q11.csv


10 - 11
```python
import pandas as pd
import warnings
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px

py.init_notebook_mode(connected=True)
warnings.filterwarnings('ignore', category=FutureWarning)

# q10 = pd.read_csv('temp_big_data-main/q10.csv', names=['community', 'count'])
q11 = pd.read_csv('temp_big_data-main/q11.csv', names=['type', 'community', 'count'])

# 10
import plotly.express as px
# fig = px.bar(q10, x='count', y='community', orientation='h', title="Counts by Community")
# fig.show()
fig = px.bar(q11[q11.type=='OFFENSE INVOLVING CHILDREN'], x='count', y='community', orientation='h', title="Counts by Community")
fig.show()

# 11
q11_new = q11[q11.community <=77] # community number is 1-77 according to doc
q11_table = q11_new.pivot(index='type', columns='community', values='count')
q11_table = q11_table.fillna(0)

fig = px.imshow(q11_table,
                color_continuous_scale=px.colors.sequential.Oranges,
                title="Crime Type & Counts by Community"
               )
fig.update_layout(
    width=1000,
    height=800,)
fig.show()
```
