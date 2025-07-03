
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
admission = pd.read_csv('ddos_dataset.csv')
print(admission.head())
print(admission.isnull().sum())
print(admission.shape)
print(admission.index)
print(admission.columns.to_frame().T)
print(admission.count().to_frame().T)
print(admission.info(verbose=True))
admission.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics ")
plt.show()
