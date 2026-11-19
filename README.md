# Smart EV Charging & T2G

Prototype Streamlit de recharge intelligente et d'injection V2G. L'onglet Conducteur calcule un plan par créneaux de 15 minutes. L'onglet Station affiche les sessions validées et simule leur réponse à un pic réseau.

## Démarrage local

Avec Python 3.10 ou une version ultérieure, depuis la racine du dépôt :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Téléchargez [Electric Vehicle Charging Patterns](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns) et placez le fichier extrait dans `data/ev_charging_patterns.csv`. Le script `python data.py` peut récupérer le dataset avec KaggleHub et afficher le dossier de téléchargement.

```bash
.venv/bin/python -m streamlit run app.py
```

Pour essayer le scénario V2G, sélectionnez une Nissan Leaf et une borne DC rapide disponible, autorisez l'injection et validez la session. Dans l'onglet Station, cliquez sur **Simuler un Pic de Demande Réseau**.
