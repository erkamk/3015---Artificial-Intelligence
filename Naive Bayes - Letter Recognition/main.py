from pstats import SortKey
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd

# veriyi yukle
letters = pd.read_csv(r"file_path")

# veriyi egitim ve test olarak ayir
train = letters.head(16000)
test = letters.tail(4000)

# bazi bilgileri kontrol amacli bastir
print(train.Letter)
print(train.values[0:10,:])
data = train.values
L = train.Letter.values
print(test.values[0:10,:])
#%%
# egitim verisini grupla ve oncul olasılıkları P(letter) hesapla
letter_num =  train.groupby('Letter')['Letter'].count()
print(letter_num)
letter_counts =  letter_num.values
letter_sum = sum(letter_counts)
prior_prob = letter_counts/letter_sum
#%%
# egitim verisini grupla ve kosullu olasilik modellerini olustur
features = train.columns
num_features = features.shape[0]

for i in range(1,num_features):
    fi =  train.groupby(['Letter', features[i]])[features[i]].count()
    fi = fi.rename_axis(['Letter', "Number"])
    print(fi.values)
    print(fi.index.values)

indices = train.Letter == 'A'
print(data[indices,:])


#%%
# Her harf için dataframe'de harf satırı (A-Z = 26'şar adet) 
# ve her harf satırı için Sayı ([0-15] arası) satırları eklendi.
df = pd.DataFrame(columns = ['Letter', 'Number'])
all_letters = list(map(chr, range(65, 91)))

for i in all_letters:
    for j in range(0,16):
        df.loc[len(df.index)] = [i,j]

#%%
# Her F sütunu, Letter ve Number sütunları üzerinden left join ile fi değişkeninden df değişkenine aktarıldı.
# fi değişkeninde 0 değerine sahip olan değerler girilmemişti. Dolayısıyla left join yapıldıktan sonra değeri
# 0 olan hücreler nan değerine sahiptir.
for i in range(1,num_features):
    fi =  train.groupby(['Letter', features[i]])[features[i]].count()
    fi = fi.rename_axis(['Letter', "Number"])
    df = df.merge(fi, on=['Letter','Number'], how='left')

#%%
# nan değerler 0 ile dolduruldu. F ile başlayan sütunlara, formül nedeniyle 1 eklendi.
df.fillna(0, inplace = True)
f_columns = []
for each in range(1,17):
    df['F'+str(each)] += 1
    f_columns.append('F'+str(each))
#%%
# Verilen formülün payda kısmı uygulandı.
for each in all_letters:
    df.loc[df.Letter.eq(each),f_columns] /= letter_num[each] + 16
#%%

def tahmin_set(test_values):
# parametreden tüm test verilerini alır. Her hücre için tahmin, tahmin_tekli() fonksiyonunda
# yapıldığı için döngü içinde parametreden aldığı satırların tahminlerini listeye kaydeder ve listeyi geri döndürür.
    tahmin_listesi = []
    counter = 0
    for i in range(len(test_values)):
        x = test_values.iloc[i]     
        x = x.to_frame().T # tahmin_tekli() fonksiyonu tek row değerine sahip dataframe'e göre yazıldığı için to_frame()
        tahmin_listesi.append(tahmin_tekli(x)) 
        counter += 1
        if counter % 100 == 0: #Ne kadarının tamamlandığı bilgisini her %2.5 değerde bastırır.
            print(counter/len(test)*100," kadarı tamamlandı.")        
    return tahmin_listesi

def tahmin_tekli(test_single_val):
    # input olarak tek elemanlı dataframe alır. Test verisindeki F sütunlarındaki değerleri
    # df'deki Number sütunundan bulur, harfin prior_prob değeri ile toplar ve her harf için tekrarlanır.
    # Max değerin harf bilgisini geri döndürülür.
    liste = []
    test_single_val.reset_index(drop = True, inplace = True)

    for ind, each in enumerate(all_letters):
        toplam = prior_prob[ind]
        x = df[df.Letter == each]

        for i in range(len(f_columns)):
            toplam += (x.loc[x.Number.eq(test_single_val.iloc[0,i+1]), f_columns[i]]).values[0]
            
        liste.append(toplam)
    
    max_val = max(liste)
    index_max_val = liste.index(max_val)
    
    return all_letters[index_max_val]

y_pred = tahmin_set(test)
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = list(test.Letter)

# aynı satır değerinde eşleşen harfler doğru kabul edilir
# eşleşen / (eşleşen/eşleşmeyen) formülü ile doğruluk değeri hesaplanır.
positive = 0
negative = 0

for i in range(len(test)):
    if y_pred[i] == y_true[i]:
        positive += 1
    else:
        negative += 1

acc_score = positive / (positive + negative)
print("Accuracy Score = %", acc_score*100)

#%%
#confusion matrix ekrana çizdirilir.

cm = confusion_matrix(y_true, y_pred)

cm_display = ConfusionMatrixDisplay(cm, display_labels = all_letters)

fig, ax = plt.subplots(figsize=(12,12))
cm_display.plot(ax=ax)

#%%
#öncül olasılık histogram
fig = plt.figure(figsize = (10, 5))
plt.bar(all_letters, prior_prob, color ='turquoise', width = 0.8)
plt.grid()
plt.xlabel("Harfler")
plt.ylabel("Öncül olasılık değerleri")
plt.title("Harflerin öncül olasılık değerleri")
plt.show()
#%%
# soncul olasılık değerleri

# A harfinin 5 değerinde özelliklere göre soncul olasılık değerleri

a5_y = df.loc[5,f_columns]
fig = plt.figure(figsize = (10, 5))
plt.bar(f_columns, a5_y, color ='orange', width = 0.8)
plt.grid()
plt.xlabel("Özellikler")
plt.ylabel("A harfinin 5 değerinde soncul olasılık değerleri")
plt.title("A harfinin 5 değerinde özelliklere göre soncul olasılık değerleri")
plt.show()
#%%
# W harfinin 3 değerinde özelliklere göre soncul olasılık değerleri

w3_y = df.loc[355,f_columns]
fig = plt.figure(figsize = (10, 5))
plt.bar(f_columns, w3_y, color ='red', width = 0.8)
plt.grid()
plt.xlabel("Özellikler")
plt.ylabel("W harfinin 3 değerinde soncul olasılık değerleri")
plt.title("W harfinin 3 değerinde özelliklere göre soncul olasılık değerleri")
plt.show()

















