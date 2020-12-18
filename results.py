import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
df = sns.load_dataset('iris')
print(df.head())
df2 = pd.DataFrame.from_dict({'acc':[0.4, 0.5, 0.6, 0.5, 0.7, 0.3, 0.6, 0.5, 0.7, 0.5]
                                    +[0.5, 0.6, 0.5, 0.6, 0.8, 0.7, 0.5, 0.8, 0.6, 0.5]
                                    +[0.5, 0.6, 0.4, 0.5, 0.5, 0.7, 0.6, 0.4, 0.5, 0.9]
                                    +[0.4, 0.7, 0.7, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7]
                                    +[0.7, 0.7, 0.6, 0.5, 0.8, 0.6, 0.8, 0.7, 0.6, 0.6]
                                    +[0.7, 0.8, 0.8, 0.7, 0.7, 0.8, 0.6, 0.6, 0.5, 0.7],
                              'type':['T1T2' for i in range(30)]+['MRF' for i in range(30)]
})
print(df2.head())

sns.set_style("whitegrid")
sns.barplot(x='type',y='acc',data=df2, palette="deep")
sns.despine(left=True)

#sns.boxplot(x=['MRF','T1T2WI'], y={'MRF':[1,2,3,4,5],'T1T2WI':[5,6,7,8,9]} ,pallete='Blues');
plt.show()
