##########################
# Required Libraries and Functions
##########################

import datetime as dt
import pandas as pd

from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


###############################################################
# Data Preparation
###############################################################
df = pd.read_csv("dataset/flo_data_20k.csv")
df.head()

df.isnull().sum()
df.describe().T

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
           "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###############################################################
# Creating the CLTV Data Structure
###############################################################

df["last_order_date"].max()  # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.total_seconds() / (24 * 3600)) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["frequency"] = cltv_df["frequency"].astype(int)
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

(cltv_df.head())
#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f              17.0000   30.5714     5.0000           187.8740
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f             209.8571  224.8571    21.0000            95.8833
# 2  69b69676-1a40-11ea-941b-000d3a38a36f              52.2857   78.8571     5.0000           117.0640
# 3  1854e56c-491f-11eb-806e-000d3a38a36f               1.5714   20.8571     2.0000            60.9850
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f              83.1429   95.4286     2.0000           104.9900

###############################################################
# BG/NBD, Establishment of Gamma-Gamma Models, Calculation of 6-month CLTV
###############################################################

# 1. BG/NBD modelini kurunuz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                           cltv_df['frequency'],
                                           cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# 2.  Fit the Gamma-Gamma model. We added the average customer leaving value to the cltv dataframe.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3. Calculate 6-month CLTV and add it to the dataframe with the name cltv.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# Divide all your customers into 4 groups (segments) according to the 6-month standardized CLTV and add the group names to the data set.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

#                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  exp_average_value     cltv cltv_segment
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f              17.0000   30.5714          5           187.8740             0.9739             1.9479           193.6327 395.7332            A
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f             209.8571  224.8571         21            95.8833             0.9832             1.9663            96.6650 199.4307            B
# 2  69b69676-1a40-11ea-941b-000d3a38a36f              52.2857   78.8571          5           117.0640             0.6706             1.3412           120.9676 170.2242            B
# 3  1854e56c-491f-11eb-806e-000d3a38a36f               1.5714   20.8571          2            60.9850             0.7004             1.4008            67.3201  98.9455            D
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f              83.1429   95.4286          2           104.9900             0.3960             0.7921           114.3251  95.0116            D
