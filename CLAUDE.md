# CLAUDE.md — Détection de chutes par squelette 2D (15.071 Projet 1)

> Brouillon de travail. Rendu : vendredi 20 novembre 2026.
> Langue de travail : français. Code, commentaires et README : anglais.

## 1. Contexte du projet

- **Problème :** les relevages de personnes âgées qui ont chuté représentent ~20 % des interventions SSUAP à la BSPP (11e compagnie). La téléassistance actuelle (bouton, médaillon) génère trop de déclenchements intempestifs, la levée de doute est faible, et on ne connaît ni l'heure de la chute ni le temps passé au sol.
- **Décideur visé :** la BSPP, ou un opérateur de téléassistance.
- **Grille du cours (15.071) :** (i) quel problème ? (ii) que peut faire l'IA ? (iii) est-ce que ça marche, contre quels benchmarks ? (iv) est-ce que ça compte, en coûts, risque, qualité de service ?
- **Principe :** un modèle simple et bien évalué vaut mieux qu'un système complexe flou.
- **Source de départ :** la note « Détection automatique de chutes par computer vision » (Polytechnique, 2025). Elle ne cite aucun dataset.

## 2. Choix technique

On travaille sur le **squelette 2D** (17 points du corps), pas sur les pixels : plus léger, tourne en local sur une caméra, meilleur pour la vie privée (on peut jeter l'image).
**YOLOv8n-pose** est un modèle déjà entraîné (sur COCO) : on l'utilise tel quel pour extraire les squelettes, on ne l'entraîne pas. Les modèles de la section 7 sont entraînés sur ces squelettes.

## 3. Démarrer (tout est automatisé)

Prérequis : Mac, `git`, `make` et [`uv`](https://docs.astral.sh/uv/) (`brew install uv`). Rien d'autre à télécharger à la main.

```bash
git clone https://github.com/Cdr-23/Fall_detection && cd Fall_detection
make setup          # crée .venv (Python 3.12) et installe requirements.txt
```

**Les poses sont déjà dans le repo** (`poses/<séquence>.parquet`, un fichier par séquence URFD, avec les labels). P2, P3 et P4 peuvent donc coder leur algo tout de suite, sans télécharger les vidéos ni relancer YOLO.

| Commande | Ce que ça fait | Qui |
|---|---|---|
| `make setup` | environnement Python | tout le monde |
| `make data` | télécharge URFD (115 Mo) et vérifie les 70 vidéos | seulement pour la démo, les visuels ou pour refaire les poses |
| `make pose` | extrait les poses avec YOLOv8n-pose (~7 min sur Mac) ; saute les séquences déjà faites | P1, si on change de modèle ou de réglage |
| `make data-gmdcsa` | télécharge GMDCSA-24 (1,1 Go), test externe | P1 seulement |

Ne pas lancer `make pose` sans nécessité : le résultat est partagé, et deux versions des poses rendraient les comparaisons incohérentes.

**Format des poses** (`poses/<seq>.parquet`, une ligne par frame) : `seq, frame, label, person, kp0_x, kp0_y, kp0_c ... kp16_x, kp16_y, kp16_c, bx1, by1, bx2, by2, img_w, img_h`.
- `frame` est numéroté à partir de 1, comme dans les CSV de labels d'URFD.
- `label` : -1 pas allongé, 0 en train de tomber, 1 allongé. NaN quand la frame n'est pas dans le CSV d'origine (certaines frames des ADL n'ont pas de label : à ignorer ou à documenter).
- `person` vaut 0 quand YOLO ne trouve personne ; les keypoints sont alors vides. C'est fréquent quand la personne est allongée (~30 % des frames).
- Points COCO (0 nez, 5-6 épaules, 11-12 hanches, 13-14 genoux, 15-16 chevilles), en pixels de l'image RGB 320×240, avec leur confiance `_c`.

**Les données brutes (pour info)**
- URFD : 70 MP4 (640×240 : profondeur à gauche, **RGB à droite**) + 2 CSV de labels. Les frames des vidéos correspondent aux labels (vérifié : 11 936 frames, 0 écart). Licence CC BY-NC-SA 4.0 : Kwolek & Kepski, 2014, https://fenix.ur.edu.pl/mkepski/ds/uf.html
- GMDCSA-24 : ~160 clips, 4 acteurs, 3 domiciles, licence CC BY 4.0, DOI 10.5281/zenodo.13354453. Structure des dossiers et des CSV pas encore inspectée.
- CAUCAFall (optionnel, 8,3 Go, pas sans accord de l'équipe) : `curl -fL -o data/raw/caucafall.zip "https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/5"`. 100 vidéos, 10 sujets, CC BY 4.0.
- `data/` est ignoré par git ; les poses (`poses/`) sont, elles, suivies.

## 4. Ce qu'on veut en sortie

**Entrée :** une vidéo.
**Sortie :** pour chaque moment de la vidéo, une des trois classes :

| Classe | Sens | Label URFD correspondant |
|---|---|---|
| `pas chute` | la personne debout, assise ou en mouvement normal | -1 |
| `chute` | la personne est en train de tomber (transition) | 0 |
| `allongé` | la personne est au sol après la chute | 1 |

Chaque algorithme doit produire ces trois classes, avec une probabilité par classe quand c'est possible.
L'alerte se déclenche sur `chute`. `allongé` sert au temps passé au sol.

## 5. Données

| Rôle | Dataset | Contenu | Où |
|---|---|---|---|
| Entraînement + validation croisée | **URFD** (caméra 0) | 70 séquences (30 chutes, 40 ADL), labels par frame | `data/raw/` (MP4, 115 Mo) |
| Test externe | **GMDCSA-24** | ~160 clips, 4 acteurs, 3 domiciles | `data/raw/gmdcsa24/` (1,1 Go) |
| Test final | **Vidéos filmées par l'équipe** | chutes et activités dans un environnement clair | fournies par l'équipe, à déposer dans `data/own/` |
| En attente | CAUCAFall | 8,3 Go, trop lourd pour le disque | décision à prendre |

- `data/` est exclu de git (`.git/info/exclude`). Chacun télécharge de son côté avec `src/download.py`.
- Les poses extraites (quelques Mo) sont partageables.
- IKAR (AGH Cracovie, 278 vidéos de chutes, sans activités du quotidien) est écarté : on n'en a pas besoin, les vidéos de l'équipe le remplacent.
- URFD : tout le monde le télécharge. GMDCSA-24 : seulement P1, qui partage les poses extraites (quelques Mo).
- Les labels n'ont pas le même format selon le dataset : une règle unique « fenêtre = chute ? » est à écrire et à documenter.
- Les datasets sont sous licence CC BY (-NC-SA pour URFD) : à citer dans le README.

## 6. Pipeline commun (déjà fixé)

1. **Pose :** YOLOv8n-pose (déjà entraîné, utilisé tel quel) → 17 points par frame, mis en cache dans `data/pose/`.
2. **Préparation :** normalisation (centre = milieu des hanches, échelle = longueur du torse), interpolation des points manquants, fenêtres de 30 frames, pas de 5.
3. **Features :** angle du torse, vitesse et accélération verticales des hanches, ratio hauteur/largeur de la boîte, hauteur de la tête, statistiques par fenêtre.

Limite connue : YOLO détecte mal une personne allongée (~70 % des frames en MP4). À dire honnêtement dans les slides.

## 7. Les algorithmes et qui fait quoi

Tous les modèles prennent les mêmes entrées (fenêtres et features) et rendent les mêmes sorties (3 classes). Sans ça, la comparaison finale est impossible.

| Qui | Tâche | Détail |
|---|---|---|
| **P1** | Fondations + modèle naïf | Téléchargement, extraction des poses, préparation, features, découpage par séquence. Modèle naïf : « jamais de chute ». |
| **P2** | Règles + régression | **Règles à seuils** (déterministe : vitesse verticale + angle du torse), puis **régression logistique** sur les features clés. |
| **P3** | Arbre + boosting | **Arbre de décision (CART)**, puis **gradient boosting** avec importance des features et SHAP. |
| **P4** | Deep learning | **Réseau de neurones** (feed-forward ou 1D-CNN) sur la séquence brute des squelettes. |
| **P5** | **À faire à la fin, tous ensemble** | **Comparaison de tous les algos** et de leurs performances entre eux (voir section 9). |

Dépendance : P2, P3 et P4 attendent que P1 ait livré les poses et les features. P1 les livre en premier.

## 8. Règles de travail

**Git :**
- **Un commit à chaque fois qu'un algo est fini d'être codé**, et un commit par étape terminée en général.
- **Aucune mention de Claude dans les commits** : pas de ligne `Co-Authored-By`, pas de « Generated with ».
- Messages courts, à l'impératif, en anglais (ex. `Add logistic regression baseline`).
- Une branche par personne (`p1-data`, `p2-rules`, ...), fusion dans `main` quand l'algo marche.
- Ne jamais commiter de données brutes ni de vidéos.

**Méthode :**
- **Découpage par séquence, jamais par frame ni par fenêtre** (`StratifiedGroupKFold`, 5 folds, groupe = séquence). Sinon fuite de données.
- **Ne jamais toucher au fold de test** pour régler un hyperparamètre. Les seuils se choisissent sur la validation.
- Les datasets de test externe (GMDCSA-24 et les vidéos de l'équipe) ne servent qu'à tester, jamais à régler quoi que ce soit.
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

## 9. P5 — Comparaison finale

À la fin, une fois tous les algos codés :
- Un tableau unique : modèle × métriques (rappel de chute, fausses alertes/h, délai, latence, taille).
- Une courbe précision/rappel par modèle sur le même graphique.
- Les meilleures features (SHAP) et 2-3 exemples d'erreurs.
- Résultats sur URFD (validation croisée) **et** sur les tests externes, pour montrer la perte de performance en changeant de décor.
- Le modèle retenu et pourquoi (le seuil d'alerte est une décision métier, fondée sur le coût d'une fausse alerte contre une chute ratée).

## 10. Test ultime

On fait tourner tous les algos sur une vidéo dans un environnement clair, et on regarde ce qui sort en termes de classification (`pas chute` / `chute` / `allongé`) pour chaque algo.

Il faut pour ça un script `src/demo.py` qui prend une vidéo et rend, pour chaque algo, la classe prédite au fil du temps.

## 11. Livrables finaux (à répartir)

- README d'une page pour un data scientist, code relançable avec un seul `make`.
- 5 à 10 slides adressées au décideur (BSPP ou opérateur de téléassistance).
- Vidéo de 2 minutes.
- Modèle d'impact (`config/impact.yaml`) : aucun chiffre inventé, chaque paramètre est sourcé ou marqué « hypothèse », avec analyse de sensibilité.

## 12. Questions encore ouvertes

- Règle exacte du label d'une fenêtre (proposition : label de la dernière frame).
- CAUCAFall : le télécharger ou non (8,3 Go).
- Qui prend l'impact, les slides et la vidéo ?
- MP4 (320×240) ou PNG (640×480) : le test montre un gain faible, on reste sur MP4 sauf mauvaise surprise.
