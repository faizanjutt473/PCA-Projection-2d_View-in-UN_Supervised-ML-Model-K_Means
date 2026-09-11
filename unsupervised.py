import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# sample data
# Elbow Method : to find  the correct number of KMeans value in model

'''data = {
    'customer':['rehman','faizan','asad','hassan','hussain'],
    'age':[25,50,45,35,27],
    'spending':[100,200,300,400,500]
}

df  = pd.DataFrame(data)

x = df[['age','spending']]

model = KMeans(n_clusters=2,random_state=42,n_init=10)
df['group'] = model.fit_predict(x)


plt.figure(figsize=(6,5))

for group in df['group'].unique():
    group_data = df[df['group']==group]
    plt.scatter(group_data['age'],group_data['spending'],label = f'group{group}')
plt.xlabel('age')
plt.ylabel('spending')
plt.title("customer segments (K_Means)")
plt.legend()
plt.grid(True)   
plt.show() 
    
print(df)'''


# PCA : Principal Components Analysis

data = {
    'age': [20,30,40,50,45,35],
    'income':[20000,30000,40000,50000,60000,70000],
    'spending': [50,60,40,30,70,80,],
    'saving' : [4000,1000,5000,6000,8000,3000]
}

df  = pd.DataFrame(data)
print(df)

scaler  = StandardScaler()
scaled = scaler.fit_transform(df)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
pca_df = pd.DataFrame(pca_result,columns=['PCA1','PCA2'])
explained_variance = pca.explained_variance_ratio_
print('variance captured by each PCA componenet ')
print(np.round(explained_variance*100,2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'],pca_df['PCA2'],color='black',s=80)
plt.title('PCA Projection (2d_view)')
plt.xlabel('PCA1 Main Pattern')
plt.ylabel('PCA2 Minor Pattern')
plt.grid(True)
plt.show()