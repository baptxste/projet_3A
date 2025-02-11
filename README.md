# projet_3A
Projet de fin d'année Centrale


# Partie 1 :
## code 
Développer différents résaux et les tester. 
- [x] LSTM 
- [x] GRU
  - [x] gauche droite
  - [x] droite gauche 
  - [ ] bi directionel  
-   Tree based  RNN ?
- [x] TPDN
  - [x] gauche droite
  - [x] droite gauche 
  - [ ] Bi -directionel 

Dataset : 
- [x] séquence de chiffres générées cf articles TPDN
- si utilisation de texte prendre un modèle pour les embbedings type ( Word2Vec ou spice)

## Résultats :

Nous utilisons une gridsearch sur les tailles d'embeddings et la taille de l'état caché sur les RNN et une gridsearch sur la taille des embeddings pour le TPDN. Pour le sens Gauche-Droite nous obtenons : 

**Pour le RNN**
| Index | emb_size | hidden_size | final_loss | final_accuracy |
|-------|---------|------------|------------|----------------|
| 0     | 8       | 8          | 1.840307   | 0.3438         |
| 1     | 8       | 16         | 1.487727   | 0.3875         |
| 2     | 8       | 32         | 1.045227   | 0.6062         |
| **3** | **8**   | **64**     | **0.146321**   | **0.9438** |
| 4     | 8       | 128        | 0.048828   | 0.9875         |
| 5     | 16      | 8          | 1.792798   | 0.3250         |
| 6     | 16      | 16         | 1.418317   | 0.4375         |
| 7     | 16      | 32         | 0.748182   | 0.7250         |
| 8     | 16      | 64         | 0.093686   | 0.9812         |
| 9     | 16      | 128        | 0.015651   | 0.9937         |
| 10    | 32      | 8          | 1.927683   | 0.2688         |
| 11    | 32      | 16         | 1.442121   | 0.4313         |
| 12    | 32      | 32         | 0.868640   | 0.6438         |
| 13    | 32      | 64         | 0.086530   | 0.9812         |
| 14    | 32      | 128        | 0.001377   | 1.0000         |
| 15    | 64      | 8          | 1.863507   | 0.3250         |
| 16    | 64      | 16         | 1.389458   | 0.4812         |
| 17    | 64      | 32         | 0.663916   | 0.7188         |
| 18    | 64      | 64         | 0.065095   | 0.9875         |
| 19    | 64      | 128        | 0.002650   | 1.0000         |

**Pour le TPDN :**
| Index | emb_size_tpdn | final_loss |
|-------|--------------|------------|
| 0     | 4            | 0.096078   |
| **1** | **8**    | **0.087358**   |
| 2     | 16           | 0.096537   |
| 3     | 32           | 0.098577   |
| 4     | 64           | 0.099614   |

A la vu de ces résultats et pour des raisons de temps d'entraînement nous utiserons le RNN emb_size=8 hidden_size=64 et le TPDN emb_size=8. Nous obtenons des résultats similaires dans le sens Droite Gauche et utiliserons les mêmes paramètres.

# Partie 2 : 

Forcer le réseau à apprendre des représentations compositionnelles en produit de tenseur. Pb :  il faut trouver des moyens d'évaluation des résultats;

Pour le TPRU en théorie il faut que les matrices de rôles et de symboles soient orthogonales, en pratique la contrainte est trop forte et souvent relaxée

# Partie 3 : 

Faire une combinaison des réseaux, ( un appris avec représentation et un sans) puis comparer les résultats.