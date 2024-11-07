import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import MultipleLocator
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

Vp = input("Wie gross ist das versicherungstechnisch notwendige Vorsorgekapital per Bilanzstichtag (Vp)? ")
Va = input("Wie gross sind die gesamten Aktiven per Bilanzstichtag zu Marktwerten bilanziert (Va)? ")
Deckungsgrad_in_Prozent_per_Bilanzstichtag = float(Va)*100/float(Vp)
print("Deckungsgrad Jahr 0: "+ str(Deckungsgrad_in_Prozent_per_Bilanzstichtag) +"%" )

min_num_aktien = 0
max_num_aktien = 100

while True:
    inputaktien = input(f"Geben Sie den gewünschten Aktienanteil zwischen  {min_num_aktien} und {max_num_aktien} % an: ")
    try:
        numaktien = int(inputaktien)
        if min_num_aktien <= numaktien <= max_num_aktien:
            break
        else:
            print(f"Nummer muss zwischen {min_num_aktien} und {max_num_aktien} liegen.")
    except ValueError:
        print("Ungültige Antwort. Bitte geben Sie eine andere Nummer ein.")

print(f"Der Aktienanteil beträgt {numaktien} %.")

min_num_immo = 0
max_num_immo = 100-float(inputaktien)

while True:
    inputimmo = input(f"Geben Sie den gewünschten Immobilienanteil zwischen  {min_num_immo} und {max_num_immo} % an: ")
    try:
        numimmo = int(inputimmo)
        if min_num_immo <= numimmo <= max_num_immo:
            break
        else:
            print(f"Nummer muss zwischen {min_num_immo} und {max_num_immo} liegen.")
    except ValueError:
        print("Ungültige Antwort. Bitte geben Sie eine andere Nummer ein.")

print(f"Der Immobilienanteil beträgt {numimmo} %.")
print("Der Anleihenanteil beträgt "+str(100-float(inputaktien)-float(inputimmo))+" %.")

ticker1 = yf.Ticker("ACWI")
hist1 = ticker1.history(period="10y")
open1 = hist1["Open"]
open1.head()
openprice1 = open1.array

f = int(len(openprice1))
results1 = np.zeros((f-252))

for i in range(f-252):
    arr1 = np.zeros(f-252)
    arr1[i] = (openprice1[i+252]-openprice1[i])/(openprice1[i])
    locals()[f"results{i}"] = arr1
    results1[i] = arr1[i]

mu1 = np.mean(results1)
sigma1 = np.std(results1)
print(mu1, sigma1)

ticker2 = yf.Ticker("RWO")
hist2 = ticker2.history(period="10y")
open2 = hist2["Open"]
open2.head()
openprice2 = open2.array

n = int(len(openprice2))
results2 = np.zeros((n-252))

for i in range(n-252):
    arr2 = np.zeros(n-252)
    arr2[i] = (openprice2[i+252]-openprice2[i])/(openprice2[i])
    locals()[f"results{i}"] = arr2
    results2[i] = arr2[i]

mu2 = np.mean(results2)
sigma2 = np.std(results2)
print(mu2, sigma2)

ticker3 = yf.Ticker("XG7S.DE")
hist3 = ticker3.history(period="10y")
open3 = hist3["Open"]
openprice3 = open3.to_list()
q = (len(open3.to_list()))-19

while len(openprice3) >= q:
    openprice3 = openprice3[:5] + openprice3[7:]

k = int(len(openprice3))
results3 = np.zeros((n-252))

for i in range(k-252):
    arr3 = np.zeros(k-252)
    arr3[i] = (openprice3[i+252]-openprice3[i])/(openprice3[i])
    locals()[f"results{i}"] = arr3
    results3[i] = arr3[i]

mu3 = np.mean(results3)
sigma3 = np.std(results3)
print(mu3, sigma3)

asset1_symbol = 'ACWI'
asset2_symbol = 'RWO'
asset3_symbol = 'XG7S.DE'
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
asset1_data = yf.download(asset1_symbol, start=start_date, end=end_date)
asset2_data = yf.download(asset2_symbol, start=start_date, end=end_date)
asset3_data = yf.download(asset3_symbol, start=start_date, end=end_date)
asset1_open = asset1_data['Open']
asset2_open = asset2_data['Open']
asset3_open = asset3_data['Open']

merged_data12 = asset1_open.to_frame().join(asset2_open.to_frame(), how='outer', lsuffix='_asset1', rsuffix='_asset2')
merged_data12.fillna(method='ffill', inplace=True)
correlation_coefficientAkIm = merged_data12['Open_asset1'].corr(merged_data12['Open_asset2'])
print(correlation_coefficientAkIm)

merged_data13 = asset1_open.to_frame().join(asset3_open.to_frame(), how='outer', lsuffix='_asset1', rsuffix='_asset2')
merged_data13.fillna(method='ffill', inplace=True)
correlation_coefficientAkAn = merged_data13['Open_asset1'].corr(merged_data13['Open_asset2'])
print(correlation_coefficientAkAn)

merged_data23 = asset2_open.to_frame().join(asset3_open.to_frame(), how='outer', lsuffix='_asset1', rsuffix='_asset2')
merged_data23.fillna(method='ffill', inplace=True)
correlation_coefficientImAn = merged_data23['Open_asset1'].corr(merged_data23['Open_asset2'])
print(correlation_coefficientImAn)

Aktienanteil = float(inputaktien)/100
Immobilienanteil = float(inputimmo)/100
Anleihenanteil = (100-float(inputaktien)-float(inputimmo))/100

mu_Aktien = mu1
sigma_Aktien = sigma1

mu_Immobilien = mu2
sigma_Immobilien = sigma2

mu_Anleihen = mu3
sigma_Anleihen = sigma3

Rendite = Aktienanteil*mu_Aktien+Immobilienanteil*mu_Immobilien+Anleihenanteil*mu_Anleihen
Varianz = (Aktienanteil**2)*sigma_Aktien**2+(Immobilienanteil**2)*sigma_Immobilien**2+(Anleihenanteil**2)*sigma_Anleihen**2+2*Aktienanteil*Immobilienanteil*correlation_coefficientAkIm*sigma_Aktien*sigma_Immobilien+2*Aktienanteil*Anleihenanteil*correlation_coefficientAkAn*sigma_Aktien*sigma_Anleihen+Immobilienanteil*Anleihenanteil*correlation_coefficientImAn*sigma_Immobilien*sigma_Anleihen

mu_gesamt = float(Rendite)
sigma_gesamt = float(Varianz)**(1/2)
random_number_gesamt = random.gauss(mu_gesamt, sigma_gesamt)
print(mu_gesamt)
print(sigma_gesamt)

n = int(input("Anzahl Jahre: "))+1
i = int(input("Anzahl Simulationen: "))

results = np.zeros((i, n))

for j in range(i):
    arr = np.zeros(n)
    for k in range(n):
        if k < 1:
            arr[0] = Deckungsgrad_in_Prozent_per_Bilanzstichtag
        else:
            random_number_gesamt = random.gauss(mu_gesamt, sigma_gesamt)
            if arr[k-1]*(1+random_number_gesamt) <= 108:
                arr[k] = arr[k-1]*(1+random_number_gesamt)-arr[k-1]*(1+random_number_gesamt)*0.01
            if 108 < arr[k-1]*(1+random_number_gesamt) <= 114:
                arr[k] = arr[k-1]*(1+random_number_gesamt)-arr[k-1]*(1+random_number_gesamt)*0.0175
            if arr[k-1]*(1+random_number_gesamt) > 114:
                arr[k] = arr[k-1]*(1+random_number_gesamt)-arr[k-1]*(1+random_number_gesamt)*0.025
        locals()[f"results{j}"] = arr
        results[j] = arr

def create_x_values(n):
    lst = [l for l in range(n)]
    return lst
x = create_x_values(n)

fig, ax = plt.subplots()

for j in range(i):
    ax.plot(x, locals()[f"results{j}"], marker="o", markersize=2, linewidth=1)

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

ax.set_xticks(x)
ax.set_xlabel('n Jahre', loc="right", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_ylabel('Deckungsgrad in %', loc="top", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_title('Deckungsgrad einer Pensionskasse nach n Jahren , '+str(i)+" Simulationen,"+"\nAnteile: Aktien: "+str(inputaktien)+" %, Immobilien: "+str(inputimmo)+"%, Anleihen: "+str(100-float(inputaktien)-float(inputimmo))+"%", pad=10, fontname="Times New Roman", fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticks(), fontname="Times New Roman")
ax.yaxis.set_major_locator(plt.FixedLocator(ax.get_yticks()))
ax.set_yticklabels(ax.get_yticks(), fontname="Times New Roman")

plt.show()

percentiles = []
for k in range(n):
    percentile_01 = np.percentile(results[:, k], 1)
    percentile_25 = np.percentile(results[:, k], 25)
    percentile_50 = np.percentile(results[:, k], 50)
    percentile_75 = np.percentile(results[:, k], 75)
    percentile_99 = np.percentile(results[:, k], 99)
    percentiles.append([percentile_01, percentile_25, percentile_50, percentile_75, percentile_99])

value = 100
rst = [value] * n

fig, ax = plt.subplots()
ax.plot(x, rst, label="100% Basislinie", linestyle="dashed")
ax.plot(x, [p[0] for p in percentiles], label='1% Perzentil', marker="o", markersize=2, linewidth=1)
ax.plot(x, [p[1] for p in percentiles], label='1. Quartil', marker="o", markersize=2, linewidth=1)
ax.plot(x, [p[2] for p in percentiles], label='2. Quartil / Median', marker="o", markersize=2, linewidth=1)
ax.plot(x, [p[3] for p in percentiles], label='3. Quartil', marker="o", markersize=2, linewidth=1)
ax.plot(x, [p[4] for p in percentiles], label='99% Perzentil', marker="o", markersize=2, linewidth=1)

ax.fill_between(x, [p[0] for p in percentiles], [p[1] for p in percentiles], color='lightblue', alpha=0.3)
ax.fill_between(x, [p[1] for p in percentiles], [p[3] for p in percentiles], color='lightblue', alpha=0.9)
ax.fill_between(x, [p[3] for p in percentiles], [p[4] for p in percentiles], color='lightblue', alpha=0.3)

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

ax.set_xticks(x)
ax.xaxis.set_major_locator(plt.FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticks(), fontname="Times New Roman")
ax.yaxis.set_major_locator(plt.FixedLocator(ax.get_yticks()))
ax.set_yticklabels(ax.get_yticks(), fontname="Times New Roman")
ax.set_xlabel('n Jahre', loc="right", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_ylabel('Deckungsgrad in %', loc="top", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_title('Deckungsgrad einer Pensionskasse nach n Jahren - in Quartilen, '+str(i)+" Simulationen,"+"\nAnteile: Aktien: "+str(inputaktien)+" %, Immobilien: "+str(inputimmo)+"%, Anleihen: "+str(100-float(inputaktien)-float(inputimmo))+"%", pad=10, fontname="Times New Roman", fontsize=15)
ax.legend(prop="Times New Roman")

step_size = 10 
locator = MultipleLocator(base=step_size)
plt.gca().yaxis.set_major_locator(locator)

plt.show()

below_100 = []
for position in range(n): 
    count = 0
    for j in range(1, i):
        lst = locals()[f"results{j}"]
        if lst[position] < 100:
            count += 1
    percentage = (count / i) * 100
    below_100.append(percentage)

fig, ax = plt.subplots()

y = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
bottom,top = plt.ylim(0, 100)

plt.plot(x, below_100, marker="o", markersize=2, linewidth=1)

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xlabel('n Jahre', loc="right", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_ylabel('Deckungsgrad unter 100%, in %', loc="top", labelpad=10, fontname="Times New Roman", fontsize=12)
ax.set_title('Deckungsgrad unter 100% nach n Jahren in % , '+str(Deckungsgrad_in_Prozent_per_Bilanzstichtag)+" % in Jahr 0"+"\nAnteile: Aktien: "+str(inputaktien)+" %, Immobilien: "+str(inputimmo)+"%, Anleihen: "+str(100-float(inputaktien)-float(inputimmo))+"%", pad=10, fontname="Times New Roman", fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(ax.get_xticks()))
ax.set_xticklabels(ax.get_xticks(), fontname="Times New Roman")
ax.yaxis.set_major_locator(plt.FixedLocator(ax.get_yticks()))
ax.set_yticklabels(ax.get_yticks(), fontname="Times New Roman")

plt.show()