# projet_3A
Projet de fin d'année Centrale


# Partie 1 :
## code 
Développer différents résaux et les tester. 
- [x] LSTM 
- [x] GRU
-   Tree based  RNN ?
- [x] TPDN
  - [x] gauche droite
  - [ ] droite gauche 

Dataset : 
- [x] séquence de chiffres générées cf articles TPDN
- si utilisation de texte prendre un modèle pour les embbedings type ( Word2Vec ou spice)

## Résultats :
# Partie 2 : 

Forcer le réseau à apprendre des représentations compositionnelles en produit de tenseur. Pb :  il faut trouver des moyens d'évaluation des résultats;

Pour le TPRU en théorie il faut que les matrices de rôles et de symboles soient orthogonales, en pratique la contrainte est trop forte et souvent relaxée

# Partie 3 : 

Faire une combinaison des réseaux, ( un appris avec représentation et un sans) puis comparer les résultats.