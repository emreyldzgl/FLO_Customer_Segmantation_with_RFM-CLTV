###############################################################
# Customer Segmentation with RFM
###############################################################
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
###############################################################
# Business Problem
###############################################################

import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("dataset/flo_data_20k.csv")

df.head()
#                            master_id    order_channel   last_order_channel  \
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile
# 2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App
# 3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop
#   first_order_date last_order_date last_order_date_online  \
# 0       2020-10-30      2021-02-26             2021-02-21
# 1       2017-02-08      2021-02-16             2021-02-16
# 2       2019-11-27      2020-11-27             2020-11-27
# 3       2021-01-06      2021-01-17             2021-01-17
# 4       2019-08-03      2021-03-07             2021-03-07
#   last_order_date_offline  order_num_total_ever_online  \
# 0              2021-02-26                        4.000
# 1              2020-01-10                       19.000
# 2              2019-12-01                        3.000
# 3              2021-01-06                        1.000
# 4              2019-08-03                        1.000
#    order_num_total_ever_offline  customer_value_total_ever_offline  \
# 0                         1.000                            139.990
# 1                         2.000                            159.970
# 2                         2.000                            189.970
# 3                         1.000                             39.990
# 4                         1.000                             49.990
#    customer_value_total_ever_online       interested_in_categories_12
# 0                           799.380                           [KADIN]
# 1                          1853.580  [ERKEK, COCUK, KADIN, AKTIFSPOR]
# 2                           395.350                    [ERKEK, KADIN]
# 3                            81.980               [AKTIFCOCUK, COCUK]
# 4                           159.990                       [AKTIFSPOR]  """"""
###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################
df.info()
# Data columns (total 12 columns):
#   Column                             Non-Null Count  Dtype
# ---  ------                             --------------  -----
# 0   master_id                          19945 non-null  object
# 1   order_channel                      19945 non-null  object
# 2   last_order_channel                 19945 non-null  object
# 3   first_order_date                   19945 non-null  object
# 4   last_order_date                    19945 non-null  object
# 5   last_order_date_online             19945 non-null  object
# 6   last_order_date_offline            19945 non-null  object
# 7   order_num_total_ever_online        19945 non-null  float64
# 8   order_num_total_ever_offline       19945 non-null  float64
# 9   customer_value_total_ever_offline  19945 non-null  float64
# 10  customer_value_total_ever_online   19945 non-null  float64
# 11  interested_in_categories_12        19945 non-null  object


df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])

df["order_num_total_OnOFF"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_total_value_OnOFF"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_online"]

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
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
rfm.head()

#                                      last_order_date  order_num_total_OnOFF  \
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f               10                  5.000
# 00034aaa-a838-11e9-a2fc-000d3a38a36f              298                  3.000
# 000be838-85df-11ea-a90b-000d3a38a36f              213                  4.000
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f               27                  7.000
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f               20                  7.000
#                                       customer_total_value_OnOFF
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f                     697.760
# 00034aaa-a838-11e9-a2fc-000d3a38a36f                     237.980
# 000be838-85df-11ea-a90b-000d3a38a36f                     713.940
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f                     685.940
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f                    2640.700

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

#              count    mean      std    min     25%     50%      75%       max
# recency   19945.000 134.458  103.281  2.000  43.000 111.000  202.000   367.000
# frequency 19945.000   5.025    4.743  2.000   3.000   4.000    6.000   202.000
# monetary  19945.000 994.643 1665.204 25.980 299.960 572.920 1156.880 90440.260

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head()

#                                      recency  frequency  monetary  \
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f       10      5.000   697.760
# 00034aaa-a838-11e9-a2fc-000d3a38a36f      298      3.000   237.980
# 000be838-85df-11ea-a90b-000d3a38a36f      213      4.000   713.940
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f       27      7.000   685.940
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f       20      7.000  2640.700
#                                     recency_score frequency_score  \
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f             5               4
# 00034aaa-a838-11e9-a2fc-000d3a38a36f             1               2
# 000be838-85df-11ea-a90b-000d3a38a36f             2               3
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f             5               4
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f             5               4
#                                     monetary_score RF_SCORE
# master_id
# 00016786-2f5a-11ea-bb80-000d3a38a36f              3       54
# 00034aaa-a838-11e9-a2fc-000d3a38a36f              1       12
# 000be838-85df-11ea-a90b-000d3a38a36f              3       23
# 000c1fe2-a8b7-11ea-8479-000d3a38a36f              3       54
# 000f5e3e-9dde-11ea-80cd-000d3a38a36f              5       54

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#                     recency       frequency       monetary
#                      mean count   |  mean count   |  mean count
# segment
# about_to_sleep      114.032  1643     2.407  1643  390.167  1643
# at_Risk             242.329  3152     4.470  3152  778.398  3152
# cant_loose          235.159  1194    10.717  1194 2228.145  1194
# champions            17.142  1920     8.965  1920 2153.748  1920
# hibernating         247.426  3589     2.391  3589  399.423  3589
# loyal_customers      82.558  3375     8.356  3375 1706.564  3375
# need_attention      113.037   806     3.739   806  616.804   806
# new_customers        17.976   673     2.000   673  390.163   673
# potential_loyalists  36.870  2925     3.311  2925  601.686  2925
# promising            58.695   668     2.000   668  352.052   668

rfm.to_csv("rfm.csv")


# TASK 1 #

# FLO is adding a new women's shoe brand to its structure. Product prices of the brand included in the general customer
# above their preferences. Therefore, for the promotion of the brand and product sales,
# it is important to work specifically with the customers in the profile that will be interested.
# to get in touch. Shopping from loyal customers (champions, loyal_customers) and women category
# are customers to be contacted specially. Save the id numbers of these customers in the csv file.


special_customers = (rfm[(rfm["segment"] == "champions") | (rfm["segment"] == "loyal_customers")])

women_categories = df[(df["interested_in_categories_12"]).str.contains("KADIN")]

special_women_customer = pd.merge(special_customers,
                                  women_categories[["interested_in_categories_12", "master_id"]], on=["master_id"])

spewc = (special_women_customer.drop(special_women_customer.loc[:, 'recency':'interested_in_categories_12'].columns, axis=1))

# TASK 2 #

# Up to 40% discount is planned for Men's and Children's products.
# In the past interested in the categories related to this discount good customers who are good customers but have not been shopping for a long time,
# customers who should not be lost, dormant customers and new customers
# incoming customers want to be specifically targeted. Save the ids of the customers in the appropriate profile to csv file.

cus_profile = rfm[(rfm["segment"] == "cant_loose") | (rfm["segment"] == "about_to_sleep") | (rfm["segment"] == "new_customers")]

man_boy_cus = df[(df["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]

man_boy_cus_profile = pd.merge(cus_profile,man_boy_cus[["interested_in_categories_12", "master_id"]], on=["master_id"])

man_boy_cus_profile = man_boy_cus_profile.drop(man_boy_cus_profile.loc[:, 'Recency':'interested_in_categories_12'].columns, axis=1)

man_boy_cus_profile.to_csv("man_customer.csv")
