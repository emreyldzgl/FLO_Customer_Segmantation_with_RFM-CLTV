import re
import pandas as pd
import plotly.express as px
import plotly.io as pio


############## TREEMAP VISUALIZATION ###################

rfm = pd.read_csv("rfm.csv")
rfm_data = pd.DataFrame(rfm[['recency_score', 'frequency_score']])

# Segment mapping
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

# Her koordinat için müşteri sayısını hesapla
customer_counts = rfm_data.groupby(['recency_score', 'frequency_score']).size().reset_index(name='customer_count')

# Her hücreyi segmentine göre etiketleyin
data = []
for i in range(1, 6):
    for j in range(1, 6):
        cell = f"{i}{j}"
        segment = next((value for pattern, value in seg_map.items() if re.match(pattern, cell)), 'unclassified')
        customer_count = customer_counts[(customer_counts['recency_score'] == i) & (customer_counts['frequency_score'] == j)]['customer_count'].sum()
        data.append({'x': j, 'y': i, 'segment': segment, 'customer_count': customer_count, 'id': cell})

# Veriyi DataFrame'e dönüştür
df2 = pd.DataFrame(data)

# Treemap oluştur
fig = px.treemap(df2, path=['segment', 'x', 'y'], values='customer_count',
                 color="customer_count", color_continuous_scale='balance',
                 title='Customer Distribution by Segment')

# Hovertemplate ve customdata ekle
fig.data[0].customdata = df2[['customer_count', 'id']].values
fig.data[0].hovertemplate = 'RFM SCORE: %{customdata[1]}<br>COUNT: %{customdata[0]}<extra></extra>'

# Her bir hücreye doğrudan metin ekle
fig.update_traces(textinfo='label+text',
                  texttemplate='%{label}<br>RFM SCORE: %{customdata[1]}<br>COUNT: %{customdata[0]}',
                  textfont=dict(size=14))

# Grafiği göster
pio.show(fig)

fig.write_html("treemap.html")




############## PIE CHART VISUALIZATION ###################

rfm = pd.read_csv("rfm.csv")

rfm_data = pd.DataFrame(rfm[['recency_score', 'frequency_score']])

# Segment mapping
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

# Her koordinat için müşteri sayısını hesapla
customer_counts = rfm_data.groupby(['recency_score', 'frequency_score']).size().reset_index(name='customer_count')

# Her hücreyi segmentine göre etiketleyin
data = []
for i in range(1, 6):
    for j in range(1, 6):
        cell = f"{i}{j}"
        segment = next((value for pattern, value in seg_map.items() if re.match(pattern, cell)), 'unclassified')
        customer_count = customer_counts[(customer_counts['recency_score'] == i) & (customer_counts['frequency_score'] == j)]['customer_count'].sum()
        data.append({'segment': segment, 'customer_count': customer_count})

# Veriyi DataFrame'e dönüştür
df = pd.DataFrame(data)

# Pasta grafiği oluştur
fig = px.pie(df, names='segment', values='customer_count', title='Customer Distribution by Segment')

# Grafiği göster
pio.show(fig)

fig.write_html("pie_chart.html")
