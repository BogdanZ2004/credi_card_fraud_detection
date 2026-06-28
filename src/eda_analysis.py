import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path, figures_dir):
    print("Učitavanje podataka za analizu...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Greška: Fajl {input_path} nije pronađen.")
        return
        
    # Osiguravamo da folder za slike postoji
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- GRAFIK 1: Prikaz nebalansiranosti klasa ---
    print("1. Generisanje grafika raspodele klasa...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='Class', data=df, palette='Set2')
    plt.title('Raspodela klasa (0: Legitimne, 1: Prevare)', fontsize=14)
    plt.yscale('log') # Logaritamska skala zbog ogromne razlike
    plt.ylabel('Broj transakcija (Log skala)')

    # Ispisivanje tačnih brojeva iznad kolona
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.35, p.get_height() + 50))
        
    class_fig_path = os.path.join(figures_dir, 'class_distribution.png')
    plt.savefig(class_fig_path, bbox_inches='tight')
    plt.close() # Zatvaramo plot da ne troši memoriju
    
    fraud_pct = (df['Class'].value_counts()[1] / len(df)) * 100
    print(f"   -> Procenat prevara: {fraud_pct:.3f}%")
    print(f"   -> Slika sačuvana u: {class_fig_path}")

    # --- GRAFIK 2: Matrica korelacije ---
    print("\n2. Generisanje matrice korelacije (ovo može potrajati par sekundi)...")
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm_r', annot=False, fmt='.2f')
    plt.title('Matrica korelacije svih atributa', fontsize=16)
    
    corr_fig_path = os.path.join(figures_dir, 'correlation_matrix.png')
    plt.savefig(corr_fig_path, bbox_inches='tight')
    plt.close()
    print(f"   -> Slika sačuvana u: {corr_fig_path}")

    # --- TEKSTUALNI IZVEŠTAJ O KORELACIJI ---
    print("\n==================================================")
    print("Atributi koji imaju najveći uticaj na detekciju prevare:")
    print("==================================================")
    # Gledamo apsolutnu vrednost korelacije (i jake pozitivne i jake negativne su bitne)
    korelacija_sa_klasom = corr['Class'].drop('Class').abs().sort_values(ascending=False)
    print(korelacija_sa_klasom.head(10))

if __name__ == "__main__":
    # Putanje pretpostavljaju da skriptu pokrećeš iz src/ foldera
    INPUT_FILE = "../data/processed/creditcard_processed.csv"
    FIGURES_DIR = "../results/figures"
    
    run_eda(INPUT_FILE, FIGURES_DIR)