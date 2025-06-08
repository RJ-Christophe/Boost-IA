import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler


def generer_donnees():
    X = []
    y = []

    for _ in range(300):
        
        X.append([
            np.random.uniform(0, 5),   
            np.random.uniform(0, 5),   
            np.random.uniform(0, 1.5), 
            np.random.uniform(6, 10),  
            np.random.choice([1, 2])   
        ])
        y.append(0)

        
        X.append([
            np.random.uniform(5, 7),   
            np.random.uniform(5, 10),  
            np.random.uniform(1.5, 2), 
            np.random.uniform(4, 6),   
            2                          
        ])
        y.append(1)

        
        X.append([
            np.random.uniform(7, 9),  
            np.random.uniform(10, 15), 
            np.random.uniform(2, 3),   
            np.random.uniform(0, 3),   
            3                          
        ])
        y.append(2)

    X = np.array(X)
    y = to_categorical(y, 3)

    
    print("Distribution des classes : {0: 300, 1: 300, 2: 300}")
    return X, y



def entrainer_modele():
    global model, scaler
    X, y = generer_donnees()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=20, verbose=0)


def evaluer_forme():
    valeurs = np.array([
        slider_sommeil.get(),
        slider_pas.get(),
        slider_eau.get(),
        slider_ecran.get(),
        slider_repas.get()
    ]).reshape(1, -1)

    valeurs_scaled = scaler.transform(valeurs)
    prediction = model.predict(valeurs_scaled, verbose=0)

    print("Valeurs entrées :", valeurs)
    print("Valeurs transformées :", valeurs_scaled)
    print("Prédiction brute :", prediction)
    print("Classe prédite :", np.argmax(prediction))

    classe = np.argmax(prediction)

    if classe == 0:
        message = " Forme faible. Repose-toi bien !"
    elif classe == 1:
        message = " Forme moyenne. Tu peux mieux faire !"
    else:
        message = " Bonne forme ! Continue comme ça !"

    messagebox.showinfo("Résultat", message)
    historique_scores.append(classe)


def reentrainer():
    entrainer_modele()
    messagebox.showinfo("Réentraînement", "L'IA a été réentraînée avec succès !")


def afficher_evolution():
    if not historique_scores:
        messagebox.showinfo("Info", "Aucune évaluation encore.")
        return

    plt.figure(figsize=(8, 4))
    plt.plot(historique_scores, marker='o', linestyle='-', color='blue')
    plt.title("Évolution de ta forme au fil du temps")
    plt.xlabel("Essai")
    plt.ylabel("Forme (0 = faible, 1 = moyenne, 2 = bonne)")
    plt.ylim(-0.1, 2.1)
    plt.grid(True)
    plt.show()


def remplir_valeurs_recommandees():
    slider_sommeil.set(8)
    slider_pas.set(12)
    slider_eau.set(2.5)
    slider_ecran.set(3)
    slider_repas.set(3)


fenetre = tk.Tk()
fenetre.title(" Coach Santé IA")
fenetre.geometry("400x500")

historique_scores = []

tk.Label(fenetre, text="Sommeil (heures)").pack()
slider_sommeil = tk.Scale(fenetre, from_=0, to=12, resolution=1, orient='horizontal')
slider_sommeil.pack()

tk.Label(fenetre, text="Pas (en milliers)").pack()
slider_pas = tk.Scale(fenetre, from_=0, to=20, resolution=1, orient='horizontal')
slider_pas.pack()

tk.Label(fenetre, text="Eau (litres)").pack()
slider_eau = tk.Scale(fenetre, from_=0, to=3, resolution=0.1, orient='horizontal')
slider_eau.pack()

tk.Label(fenetre, text="Écran (heures)").pack()
slider_ecran = tk.Scale(fenetre, from_=0, to=10, resolution=1, orient='horizontal')
slider_ecran.pack()

tk.Label(fenetre, text="Repas / jour").pack()
slider_repas = tk.Scale(fenetre, from_=1, to=3, resolution=1, orient='horizontal')
slider_repas.pack()

tk.Button(fenetre, text=" Évaluer ma forme", command=evaluer_forme).pack(pady=5)
tk.Button(fenetre, text=" Réentraîner l'IA", command=reentrainer).pack(pady=5)
tk.Button(fenetre, text=" Afficher évolution", command=afficher_evolution).pack(pady=5)
tk.Button(fenetre, text=" Valeurs recommandées", command=remplir_valeurs_recommandees).pack(pady=5)

entrainer_modele()
fenetre.mainloop()
