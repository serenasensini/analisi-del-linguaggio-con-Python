#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt;
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 700)

consumer_complaint = pd.read_csv("consumer_complaints.csv", encoding='utf8', sep=',', parse_dates=True,low_memory=False)

sizes = consumer_complaint.groupby('sub_issue').size()

plt.rcdefaults()
fig, ax = plt.subplots()

data = consumer_complaint['sub_issue'].unique()[1:15]
y_pos = np.arange(len(data))
performance = consumer_complaint.groupby('sub_issue').size()[1:15]
error = np.random.rand(len(data))

ax.barh(y_pos, performance, xerr=error, align='center', linewidth=1000, color='#45ad80')
ax.set_yticks(y_pos)
ax.set_yticklabels(data[:15])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_title('Numero di reclami per tipologia di problema')

plt.show()

consumer_complaint.rename(columns={'consumer_disputed?':'consumer_disputed'}, inplace=True)

complain_flag = '1'*len(consumer_complaint['state'])
consumer_complaint['complains'] = [ int(x) for x in complain_flag if not x  == ',']
consumer_complaint_state_wise = consumer_complaint.groupby('state').aggregate(np.sum)
consumer_complaint_state_wise.drop('complaint_id', axis = 1, inplace =  True)
consumer_complaint_state_wise.plot(kind = 'bar')

# Stati con il massimo numero di reclami

consumer_complaint_state_wise[consumer_complaint_state_wise['complains'] == consumer_complaint_state_wise['complains'].max()]

# Stati con il minimo numero di reclami

consumer_complaint_state_wise[consumer_complaint_state_wise['complains'] == consumer_complaint_state_wise['complains'].min()]

# Società con il maggior numero di reclami
consumer_complaints = consumer_complaint.groupby('company').aggregate(np.sum)
consumer_complaints.drop('complaint_id', axis = 1, inplace =  True)

consumer_complaints[consumer_complaints['complains'] == consumer_complaints['complains'].max()]

consumer_complaint_product_wise = consumer_complaint.groupby('product').aggregate(np.sum)
consumer_complaint_product_wise.drop('complaint_id', axis = 1, inplace =  True)

# Prodotti o servizi con il maggior numero di reclami

consumer_complaint_product_wise[consumer_complaint_product_wise['complains'] == consumer_complaint_product_wise['complains'].max()]

consumer_complaint_product_wise.plot(kind = 'bar')

# Società con il miglior servizio clienti

best_cc =  consumer_complaint[(consumer_complaint.timely_response == 'Yes') &
                              (consumer_complaint.consumer_disputed == 'No')]
best_cc = best_cc.groupby('company').aggregate(np.sum)
best_cc.drop('complaint_id', axis = 1, inplace =  True)
best_cc[best_cc['complains'] == best_cc['complains'].max()]


best_cc['percent_resolution'].plot(kind ='bar')
