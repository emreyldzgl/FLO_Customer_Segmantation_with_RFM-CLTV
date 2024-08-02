import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("dataset/flo_data_20k.csv")


df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df["order_num_total_OnOFF"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_total_value_OnOFF"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]

###############################################################
# 3. Calculating RFM Metrics
###############################################################
df["last_order_date"].max()
# Timestamp('2021-05-30 00:00:00')
today_date = dt.datetime(2021, 6, 1)


rfm = df.groupby('master_id').agg({'last_order_date':
                                  lambda last_order_date: (today_date - last_order_date.max()).days,
                                   "order_num_total_OnOFF":
                                  lambda order_num_total_OnOFF: order_num_total_OnOFF,
                                   "customer_total_value_OnOFF":
                                  lambda customer_total_value_OnOFF: customer_total_value_OnOFF.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()

sns.histplot(rfm["recency"], kde=True)
plt.show()

sns.histplot(rfm["frequency"], kde=True)
plt.show()

sns.histplot(rfm["monetary"], kde=True)
plt.show()

MMScaler = MinMaxScaler()
x_scaled = MMScaler.fit_transform(rfm)
scaled_data = pd.DataFrame(x_scaled)

scaled_data.columns = ['recency', 'frequency', 'monetary']
scaled_data.head()
scaled_data.describe()


# K-MEANS CLUSTER - ELBOW METHOD
plt.figure(figsize=(10, 6))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Silhouette Score
inertia_list = []
silhouette_score_list = []

for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(scaled_data)
    silhouette_score_list.append(silhouette_score(scaled_data, kmeans.labels_))
    print(silhouette_score_list)

# [0.6291345699937643]
# [0.6291345699937643, 0.5911172490241166]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357, 0.5596430947652099]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357, 0.5596430947652099, 0.5400291299561598]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357, 0.5596430947652099, 0.5400291299561598, 0.5103877071224279]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357, 0.5596430947652099, 0.5400291299561598, 0.5103877071224279, 0.49418641318306905]
# [0.6291345699937643, 0.5911172490241166, 0.5749793169413357, 0.5596430947652099, 0.5400291299561598, 0.5103877071224279, 0.49418641318306905, 0.4644249612275098]



# K-MEANS CLUSTER (4) - ELBOW
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)
kmeans.fit(scaled_data)
pred = kmeans.predict(scaled_data)

d_frame = pd.DataFrame(rfm)
d_frame["Cluster"] = pred
d_frame.head()

#                                       recency  frequency  monetary  Cluster
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f       10      5.000   697.760        0
# 00034aaa-a838-11e9-a2fc-000d3a38a36f      298      3.000   237.980        1
# 000be838-85df-11ea-a90b-000d3a38a36f      213      4.000   713.940        2
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f       27      7.000   685.940        0
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f       20      7.000  2640.700        0

d_frame["Cluster"].value_counts()

# Cluster
# 0    7012
# 3    5462
# 2    4464
# 1    3007

d_frame.groupby("Cluster").mean()

#         recency  frequency  monetary
# Cluster
# 0         31.209      5.677  1236.002
# 1        318.433      4.183   718.719
# 2        202.782      4.624   889.930
# 3        109.885      4.978   922.278

# CUSTOMER SEGMENTATION

# 0 -> PLATINUM
# 3 -> GOLD
# 2 -> SILVER
# 1 -> BRONZE
