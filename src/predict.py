import pandas as pd
import joblib
import os

def simulate_real_time_prediction(model_path, test_data_path):
    print("1. Inicijalizacija sistema za detekciju prevara...")
    
    # Učitavanje tvog najboljeg modela
    if not os.path.exists(model_path):
        print(f"Greška: Model nije pronađen na putanji {model_path}")
        return
        
    model = joblib.load(model_path)
    print("   [Sistem je onlajn. Random Forest model je učitan.]")

    # Učitavanje test podataka (glumimo bazu iz koje stižu transakcije)
    df = pd.read_csv(test_data_path)
    
    # Nalazimo jednu legitimnu i jednu prevarantsku transakciju za demonstraciju
    legitimna_transakcija = df[df['Class'] == 0].sample(1, random_state=42)
    prevara_transakcija = df[df['Class'] == 1].sample(1, random_state=42)
    
    # Pravimo "mini red vožnje" transakcija koje stižu u banku
    nove_transakcije = pd.concat([legitimna_transakcija, prevara_transakcija])
    
    # Sklanjamo 'Class' kolonu (jer u realnosti ne znamo da li je prevara pre nego što procenimo)
    X_nove = nove_transakcije.drop('Class', axis=1)
    stvarne_vrednosti = nove_transakcije['Class'].values

    print("\n2. Pristižu nove transakcije na proveru...")
    
    # Pravimo predikcije
    predikcije = model.predict(X_nove)
    verovatnoce = model.predict_proba(X_nove)[:, 1] # Verovatnoća da je prevara

    # Ispisivanje rezultata onako kako bi to video operater u banci
    print("\n==================================================")
    print("          IZVEŠTAJ O TRANSAKCIJAMA")
    print("==================================================")
    
    for i in range(len(predikcije)):
        print(f"\nTransakcija #{i+1}:")
        
        # Prikazujemo par bitnih karakteristika (Sat i Iznos)
        sat = int(X_nove.iloc[i]['Hour'])
        print(f" -> Vreme transakcije: {sat}:00 h")
        
        # Odlučivanje sistema
        if predikcije[i] == 1:
            print(f" -> ODLUKA SISTEMA: 🚨 ODBIJENO! (Sumnja na prevaru)")
            print(f" -> Sigurnost modela: {verovatnoce[i]*100:.1f}%")
        else:
            print(f" -> ODLUKA SISTEMA: ✅ ODOBRENO! (Legitimna transakcija)")
            print(f" -> Sigurnost modela: {(1 - verovatnoce[i])*100:.1f}%")
            
        # Provera sa stvarnim stanjem (za tvoju demonstraciju asistentu)
        status = "PREVARA" if stvarne_vrednosti[i] == 1 else "LEGITIMNA"
        print(f" -> (Stvarni status iz baze: {status})")

if __name__ == "__main__":
    # Putanja do najboljeg modela
    MODEL_PATH = "../models/RandomForest_100.pkl"
    TEST_DATA = "../data/processed/test_set.csv"
    
    simulate_real_time_prediction(MODEL_PATH, TEST_DATA)