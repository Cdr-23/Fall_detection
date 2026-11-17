# CLAUDE.md — Détection de chutes par squelette 2D (15.071 Projet 1)

> Brouillon de travail. Rendu : vendredi 20 novembre 2026.
> Langue de travail : français. Code, commentaires et README : anglais.

## 1. Ce qu'on veut en sortie

**Entrée :** une vidéo.
**Sortie :** pour chaque moment de la vidéo, une des trois classes :

| Classe | Sens | Label URFD correspondant |
|---|---|---|
| `pas chute` | la personne debout, assise ou en mouvement normal | -1 |
| `chute` | la personne est en train de tomber (transition) | 0 |
| `allongé` | la personne est au sol après la chute | 1 |

Chaque algorithme doit produire ces trois classes, avec une probabilité par classe quand c'est possible.
L'alerte se déclenche sur `chute`. `allongé` sert au temps passé au sol.

## 2. Données

| Rôle | Dataset | Contenu | Où |
|---|---|---|---|
| Entraînement + validation croisée | **URFD** (caméra 0) | 70 séquences (30 chutes, 40 ADL), labels par frame | `data/raw/` (MP4, 115 Mo) |
| Test externe | **GMDCSA-24** | ~160 clips, 4 acteurs, 3 domiciles | `data/raw/gmdcsa24/` (1,1 Go) |
| Test externe (chutes seulement) | **IKAR** (AGH Cracovie) | 278 vidéos de chutes, 7 sujets, boîtes englobantes | à télécharger à la main (formulaire) |
| En attente | CAUCAFall | 8,3 Go, trop lourd pour le disque | décision à prendre |

- `data/` est exclu de git (`.git/info/exclude`). Chacun télécharge de son côté avec `src/download.py`.
- Les poses extraites (quelques Mo) sont partageables.
- IKAR n'a pas d'activités du quotidien : il mesure le rappel, pas les fausses alertes.
- Les labels n'ont pas le même format selon le dataset : une règle unique « fenêtre = chute ? » est à écrire et à documenter.
- Les datasets sont sous licence CC BY (-NC-SA pour URFD) : à citer dans le README.

## 3. Pipeline commun (déjà fixé)

1. **Pose :** YOLOv8n-pose (déjà entraîné, utilisé tel quel) → 17 points par frame, mis en cache dans `data/pose/`.
2. **Préparation :** normalisation (centre = milieu des hanches, échelle = longueur du torse), interpolation des points manquants, fenêtres de 30 frames, pas de 5.
3. **Features :** angle du torse, vitesse et accélération verticales des hanches, ratio hauteur/largeur de la boîte, hauteur de la tête, statistiques par fenêtre.

Limite connue : YOLO détecte mal une personne allongée (~70 % des frames en MP4). À dire honnêtement dans les slides.

## 4. Les algorithmes et qui fait quoi

Tous les modèles prennent les mêmes entrées (fenêtres et features) et rendent les mêmes sorties (3 classes). Sans ça, la comparaison finale est impossible.

| Qui | Tâche | Détail |
|---|---|---|
| **P1** | Fondations + modèle naïf | Téléchargement, extraction des poses, préparation, features, découpage par séquence. Modèle naïf : « jamais de chute ». |
| **P2** | Règles + régression | **Règles à seuils** (déterministe : vitesse verticale + angle du torse), puis **régression logistique** sur les features clés. |
| **P3** | Arbre + boosting | **Arbre de décision (CART)**, puis **gradient boosting** avec importance des features et SHAP. |
| **P4** | Deep learning | **Réseau de neurones** (feed-forward ou 1D-CNN) sur la séquence brute des squelettes. |
| **P5** | **À faire à la fin, tous ensemble** | **Comparaison de tous les algos** et de leurs performances entre eux (voir section 6). |

Dépendance : P2, P3 et P4 attendent que P1 ait livré les poses et les features. P1 les livre en premier.

## 5. Règles de travail

**Git :**
- **Un commit à chaque fois qu'un algo est fini d'être codé**, et un commit par étape terminée en général.
- **Aucune mention de Claude dans les commits** : pas de ligne `Co-Authored-By`, pas de « Generated with ».
- Messages courts, à l'impératif, en anglais (ex. `Add logistic regression baseline`).
- Une branche par personne (`p1-data`, `p2-rules`, ...), fusion dans `main` quand l'algo marche.
- Ne jamais commiter de données brutes ni de vidéos.

**Méthode :**
- **Découpage par séquence, jamais par frame ni par fenêtre** (`StratifiedGroupKFold`, 5 folds, groupe = séquence). Sinon fuite de données.
- **Ne jamais toucher au fold de test** pour régler un hyperparamètre. Les seuils se choisissent sur la validation.
- Les datasets de test externe (GMDCSA-24, IKAR) ne servent qu'à tester, jamais à régler quoi que ce soit.
- Seed fixe partout (dans `config/params.yaml`).
- Si un score paraît trop beau (AUC proche de 1), chercher une fuite avant de le croire.
- 70 séquences, c'est petit : le dire honnêtement.
- Chaque algo sauvegarde ses résultats dans `results/<algo>.csv` avec les mêmes colonnes.
- Prévenir l'équipe avant tout choix qui change le scope.

**Mesures communes à tous les algos :**
- Au niveau fenêtre : ROC-AUC, PR-AUC, rappel, précision, F1.
- Au niveau événement : chute détectée oui/non, délai de détection (secondes), fausses alertes par heure d'ADL.
- Latence d'inférence par frame et taille du modèle.
- Moyenne ± écart-type sur les folds.

## 6. P5 — Comparaison finale

À la fin, une fois tous les algos codés :
- Un tableau unique : modèle × métriques (rappel de chute, fausses alertes/h, délai, latence, taille).
- Une courbe précision/rappel par modèle sur le même graphique.
- Les meilleures features (SHAP) et 2-3 exemples d'erreurs.
- Résultats sur URFD (validation croisée) **et** sur les tests externes, pour montrer la perte de performance en changeant de décor.
- Le modèle retenu et pourquoi (le seuil d'alerte est une décision métier, fondée sur le coût d'une fausse alerte contre une chute ratée).

## 7. Test ultime

On donne à Claude **une vidéo dans un environnement clair**. Les algos tournent dessus, puis Claude juge ce qui est correct en regardant des frames extraites de la vidéo et en les comparant aux sorties de chaque algo (`pas chute` / `chute` / `allongé`).

Il faut pour ça un script `src/demo.py` qui prend une vidéo et rend, pour chaque algo, la classe prédite au fil du temps.

## 8. Livrables finaux (à répartir)

- README d'une page pour un data scientist, code relançable avec un seul `make`.
- 5 à 10 slides adressées au décideur (BSPP ou opérateur de téléassistance).
- Vidéo de 2 minutes.
- Modèle d'impact (`config/impact.yaml`) : aucun chiffre inventé, chaque paramètre est sourcé ou marqué « hypothèse », avec analyse de sensibilité.

## 9. Questions encore ouvertes

- Règle exacte du label d'une fenêtre (proposition : label de la dernière frame).
- CAUCAFall : le télécharger ou non (8,3 Go).
- Qui prend l'impact, les slides et la vidéo ?
- MP4 (320×240) ou PNG (640×480) : le test montre un gain faible, on reste sur MP4 sauf mauvaise surprise.
