# Implied Volatility Smile & Surface Toolkit

Ce dossier propose un mini-projet complet pour calculer la volatilité implicite
à partir d'un prix d'option, construire des smiles de volatilité et interpoler
une surface de vol en fonction du strike et de la maturité. Les options sont
supposées européennes et pricées sous le modèle de Black–Scholes avec un taux
sans risque **continu**.

## Installation rapide

```bash
pip install numpy pandas matplotlib scipy
```

## Structure

- `black_scholes.py` : prix Black–Scholes et termes intermédiaires `d1`, `d2`.
- `iv_solver.py` : solveurs de volatilité implicite (Newton puis bissection).
- `data_loader.py` : chargement d'un CSV ou génération de données factices.
- `vol_smile.py` : calcul des IV pour une maturité ou pour tout le dataset.
- `vol_surface.py` : interpolation d'une surface de vol à partir des points IV.
- `visuals.py` : fonctions de visualisation (smiles 2D, surface 3D, heatmap).
- `demo_smile.py` : script de démonstration pour tracer quelques smiles.
- `demo_surface.py` : script de démonstration pour tracer la surface.

## Format de données attendu

Le CSV (option chain) doit contenir au minimum les colonnes :

- `underlying_price` : prix spot du sous-jacent
- `strike` : strike de l'option
- `maturity` : maturité en années (float)
- `rate` : taux sans risque annuel (continu)
- `option_price` : prix d'option observé (mid ou last)
- `option_type` : `call` ou `put`

Exemple minimal :

```csv
underlying_price,strike,maturity,rate,option_price,option_type
100,95,0.5,0.01,7.2,call
100,105,0.5,0.01,5.1,call
100,95,0.5,0.01,2.7,put
100,105,0.5,0.01,4.8,put
```

Si aucun fichier `option_chain.csv` n'est présent, les scripts de démo
utilisent `generate_mock_data` pour produire un jeu de données synthétique
(quadrique en strike avec une légère term structure) afin d'illustrer le
pipeline.

## Lancer les démos

Depuis la racine du dépôt :

```bash
python -m implied_vol_surface.demo_smile
python -m implied_vol_surface.demo_surface
```

Les figures apparaîtront dans des fenêtres Matplotlib (`plt.show()`).

## Points clés numériques

- Le solveur Newton–Raphson est tenté en premier pour la vitesse ;
  en cas d'échec, la bissection (toujours convergente si le signe change)
  prend le relais.
- Les volatilités sont bornées entre `1e-4` et `5` pour éviter des résultats
  aberrants.
- L'interpolation de la surface utilise `scipy.interpolate.griddata` en mode
  linéaire, avec un comblement nearest-neighbor pour les éventuels trous.

## Licence

Projet fourni à titre éducatif ; adapter avant toute utilisation en production.
