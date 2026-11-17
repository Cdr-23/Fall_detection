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

Prérequis : Mac (testé sur Apple Silicon), `git`, `make` et [`uv`](https://docs.astral.sh/uv/) (`brew install uv`), et un accès en écriture au repo GitHub (à demander au propriétaire). Rien d'autre à télécharger à la main : `uv` installe Python 3.12 tout seul.

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
- `label` (colonne URFD) : -1 pas allongé, 1 allongé, 0 pose temporaire (« en train de tomber » dans les séquences `fall-*`, mais aussi s'asseoir ou se pencher dans les `adl-*`). Le passage aux 3 classes est défini en 7.0. NaN quand la frame n'est pas dans le CSV d'origine (certaines frames des ADL n'ont pas de label : à ignorer ou à documenter).
- `person` vaut 0 quand YOLO ne trouve personne ; les keypoints sont alors vides. C'est fréquent quand la personne est allongée (~30 % des frames).
- Points COCO (0 nez, 5-6 épaules, 11-12 hanches, 13-14 genoux, 15-16 chevilles), en pixels de l'image RGB 320×240, avec leur confiance `_c`.

**Les données brutes (pour info)**
- URFD : 70 MP4 (640×240 : profondeur à gauche, **RGB à droite**) + 2 CSV de labels. Les frames des vidéos correspondent aux labels (vérifié : 11 936 frames, 0 écart). Licence CC BY-NC-SA 4.0 : Kwolek & Kepski, 2014, https://fenix.ur.edu.pl/mkepski/ds/uf.html
- GMDCSA-24 (`make data-gmdcsa`, télécharge seulement le zip ; ensuite `cd data/raw/gmdcsa24 && unzip -q GMDCSA24-v2.1.zip`) : 160 clips MP4 (79 chutes, 81 ADL), 4 sujets, 3 domiciles, licence CC BY 4.0, DOI 10.5281/zenodo.13354453. Vérifié : le zip est intact. Structure : `<dossier>/Subject N/Fall/NN.mp4`, `Subject N/ADL/NN.mp4`, plus `Subject N/Fall.csv` et `ADL.csv`. Les vidéos sont en **1280×720, ~30 fps, image RGB complète** (pas de profondeur : ne pas recadrer comme pour URFD). Les CSV ont les colonnes `File Name, Length (seconds), Time of Recording, Attire, Description, Classes` ; `Classes` donne des intervalles en secondes, par ex. `Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]` ou `Sitting[0 to 1]; Sleeping[2 to 8]`. Attention : « Falling » court jusqu'à la fin du clip (chute + sol), donc c'est plus grossier que les labels d'URFD. `src/extract_pose.py` ne gère pas ce format (il lit les MP4 d'URFD et garde la moitié droite) : il faut le généraliser avant de s'en servir.
- CAUCAFall (optionnel, 8,3 Go, pas sans accord de l'équipe) : `curl -fL -o data/raw/caucafall.zip "https://data.mendeley.com/public-api/zip/7w7fccy7ky/download/5"`. 100 vidéos, 10 sujets, CC BY 4.0.
- `data/` est ignoré par git ; les poses (`poses/`) sont, elles, suivies.

## 4. Ce qu'on veut en sortie

**Entrée :** une vidéo.
**Sortie :** pour chaque moment de la vidéo, une des trois classes :

| Classe | Sens | Label URFD correspondant |
|---|---|---|
| `pas chute` | la personne debout, assise ou en mouvement normal | -1, et aussi 0 dans les séquences `adl-*` (pose temporaire : s'asseoir, se pencher) |
| `chute` | la personne est en train de tomber (transition) | 0 dans les séquences `fall-*` uniquement (30 frames = 1 s par chute) |
| `allongé` | la personne est allongée | 1 (après une chute, ou volontairement dans les ADL : lit, canapé) |

Chaque algorithme doit produire ces trois classes, avec une probabilité par classe quand c'est possible.
L'alerte se déclenche sur `chute`. `allongé` sert au temps passé au sol.

## 5. Données

| Rôle | Dataset | Contenu | Où |
|---|---|---|---|
| Entraînement + validation croisée | **URFD** (caméra 0) | 70 séquences (30 chutes, 40 ADL), labels par frame | `data/raw/` (MP4, 115 Mo) |
| Test externe | **GMDCSA-24** | ~160 clips, 4 acteurs, 3 domiciles | `data/raw/gmdcsa24/` (1,1 Go) |
| Test final | **Vidéos filmées par l'équipe** | chutes et activités dans un environnement clair | `data/own/` (voir format ci-dessous) |
| En attente | CAUCAFall | 8,3 Go, trop lourd pour le disque | décision à prendre |

- `data/` est exclu de git (`.gitignore`). Les données brutes se téléchargent avec `make data` (voir section 3). Les poses extraites (`poses/`, 6 Mo) sont suivies par git.
- IKAR (AGH Cracovie, 278 vidéos de chutes, sans activités du quotidien) est écarté : on n'en a pas besoin, les vidéos de l'équipe le remplacent.
- **Personne n'a besoin de télécharger quoi que ce soit pour coder P2, P3, P4** : les poses sont dans le repo. `make data` (URFD, 115 Mo) ne sert qu'à la démo et aux visuels ; GMDCSA-24 (1,1 Go) ne sert qu'au test externe.
- Les labels n'ont pas le même format selon le dataset : une règle unique « fenêtre = chute ? » est à écrire et à documenter.
- Les datasets sont sous licence CC BY (-NC-SA pour URFD) : à citer dans le README.

**Vidéos de l'équipe (proposition de format) :** fichiers `data/own/fall-01.mp4`, `adl-01.mp4`, ... en ~30 fps, personne entière dans le cadre, caméra fixe, au moins une chute jouée sur tapis et des activités normales (s'asseoir, se pencher, s'allonger volontairement). Un fichier `data/own/labels.csv` avec les colonnes `file, classes`, où `classes` suit le style de GMDCSA-24 : `Falling[3.4 to 4.4]; Lying[4.4 to 10]`. Sans labels, on peut quand même regarder la sortie des algos, mais pas calculer de métriques. Ces vidéos ne servent qu'à tester.

## 6. Pipeline commun (déjà fixé)

1. **Pose :** YOLOv8n-pose (déjà entraîné, utilisé tel quel) → 17 points par frame, mis en cache dans `poses/`.
2. **Préparation :** normalisation (centre = milieu des hanches, échelle = longueur du torse), interpolation des points manquants, fenêtres de 30 frames, pas de 5.
3. **Features :** angle du torse, vitesse et accélération verticales des hanches, ratio hauteur/largeur de la boîte, hauteur de la tête, statistiques par fenêtre.

Limite connue : YOLO détecte mal une personne allongée (une personne trouvée sur 65 % des frames allongées, contre 95 % des frames debout). À dire honnêtement dans les slides.

## 7. Les algorithmes et qui fait quoi

Tous les modèles prennent les mêmes entrées et rendent les mêmes sorties (3 classes). Sans ça, la comparaison finale est impossible.

| Qui | Tâche | Branche |
|---|---|---|
| **P1** | Fondations : fenêtres, features, folds, métriques, modèle naïf | `p1-foundations` |
| **P2** | Règles à seuils, puis régression logistique | `p2-rules-logreg` |
| **P3** | Arbre de décision (CART), puis gradient boosting + SHAP | `p3-trees` |
| **P4** | Réseau de neurones sur les squelettes bruts | `p4-nn` |
| **P5** | **À faire à la fin, tous ensemble** : comparaison de tous les algos | `p5-compare` |

**Si tu es un Claude lancé par Pn : ne modifie que les fichiers de ta partie, sur ta branche. Si tu as besoin d'un changement chez un autre, dis-le à ton utilisateur.**

**Planning (mardi 17 au vendredi 20 nov.) :**
- Mardi soir : poses faites (c'est fait). P1 commence.
- **Mercredi matin : P1 pousse le contrat de données (7.0) avec quelques lignes réelles**, pour que P2, P3, P4 puissent coder sans attendre. Mercredi après-midi : P1 pousse les fichiers complets.
- Mercredi soir / jeudi midi : P2, P3, P4 poussent leurs algos et leurs résultats.
- Jeudi : P5 (comparaison, analyse d'erreurs), impact, figures.
- Vendredi : démo, slides, vidéo, relecture, rendu.

### 7.0 Contrat commun (P1 le produit, tout le monde le respecte)

**Classes** (entiers) : `0 = pas chute`, `1 = chute`, `2 = allongé`. Règle de passage depuis les labels URFD (colonne `label`, voir section 3) :
- `1` (chute) : label 0 dans une séquence `fall-*` (exactement 30 frames par chute).
- `2` (allongé) : label 1, dans n'importe quelle séquence.
- `0` (pas chute) : label -1, et label 0 dans une séquence `adl-*`. Ces frames-là (s'asseoir vite, se pencher) sont les cas difficiles à ne pas confondre avec une chute.
- Frames sans label (NaN) : ignorées.

URFD dit lui-même que le label 0 est une « pose temporaire » et ne l'utilise pas pour classer : notre règle est donc un choix d'équipe, à documenter dans les slides.

**Label d'une fenêtre** = la classe de **sa dernière frame** (le modèle dit ce qui se passe maintenant, ce qui marche aussi en temps réel). Les fenêtres dont la dernière frame n'a pas de label sont supprimées. Si P1 veut changer cette règle, il prévient les autres.

**Fichiers produits par P1 dans `processed/`** (suivis par git, petits) :

| Fichier | Contenu |
|---|---|
| `windows_meta.parquet` | une ligne par fenêtre : `window_id, seq, start_frame, end_frame, y (0/1/2), fold (0-4), kind (fall/adl), valid_frac` |
| `features.parquet` | `window_id` + les colonnes de features (liste en 7.1) |
| `X_seq.npy` | `float32 (N, 30, 17, 2)` : keypoints normalisés (x, y), pour le réseau de neurones |
| `valid.npy` | `bool (N, 30)` : `True` si la frame avait une vraie détection (pas interpolée) |
| `folds.csv` | `seq, fold` : le découpage commun à tous |

L'ordre des lignes est le même dans `windows_meta`, `features`, `X_seq` et `valid`.

**Découpage (identique pour tous) :** 5 folds, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, groupe = séquence, stratifié sur `kind` (fall / adl). **Jamais de découpage par fenêtre ou par frame.** Pour régler un hyperparamètre, on fait un découpage interne **par séquence à l'intérieur des folds d'entraînement** (`GroupKFold`), jamais sur le fold de test.

**Sorties d'un algo (obligatoires) :**
- `results/<algo>_oof.parquet` : une ligne par fenêtre, prédite quand son fold est le fold de test. Colonnes : `window_id, seq, fold, end_frame, y_true, p0, p1, p2, y_pred` (`p0/p1/p2` = probabilités de `pas chute / chute / allongé`, qui somment à 1 ; `y_pred` = argmax). Un modèle à règles met des probabilités 0 ou 1.
- `models/<algo>.joblib` (ou `.pt`) : le modèle final, entraîné sur les 70 séquences.
- Dans `src/run_<algo>.py`, une fonction `predict_proba(feats: pd.DataFrame, X_seq: np.ndarray) -> np.ndarray` de forme `(N, 3)` qui charge ce modèle. C'est ce qui permet à la démo de l'utiliser.
- Un commit dès que l'algo tourne (voir section 8).

**Alerte :** on déclenche l'alerte quand `p1 >= tau`. Par défaut `tau = 0.5` ; P5 balaie `tau`. Le seuil n'est jamais réglé sur le fold de test.

**Métriques** (calculées par `src/metrics.py`, écrit par P1, identiques pour tous) :
- Fenêtres : ROC-AUC et PR-AUC de la classe `chute`, rappel, précision, F1 par classe, matrice de confusion 3×3.
- Événements : une chute réelle = les 30 frames de label 0 d'une séquence `fall-*` (début `s0`). Elle est **détectée** si une fenêtre qui se termine entre `s0` et `s0 + 44` (la phase + 0,5 s) est prédite `chute` avec `p1 >= tau`. **Délai** = (frame de fin de la première fenêtre alertée − `s0`) / 30, en secondes (peut être négatif). Une **fausse alerte** est une fenêtre alertée hors de ces zones ; les fausses alertes à moins de 60 frames (2 s) l'une de l'autre comptent pour une seule. **Fausses alertes par heure** = nombre dans les séquences `adl-*` divisé par la durée d'ADL.
- Attention : les 40 ADL ne font que **0,083 h (5 minutes)** et les 30 chutes **0,028 h**. Les « fausses alertes par heure » sont donc très imprécises : donner le nombre brut à côté du taux, et dire honnêtement que 70 séquences c'est petit.
- Latence d'inférence par fenêtre (ms) et taille du modèle (Ko).

### 7.1 P1 — Fondations (branche `p1-foundations`)

Déjà fait : téléchargement (`make data`), extraction des poses (`make pose`), fichiers `poses/*.parquet`. Reste, dans cet ordre :

1. **Contrat d'abord (mercredi matin).** Pousse un `processed/` partiel (3 ou 4 séquences) avec les bonnes colonnes et les bons types, pour débloquer P2, P3, P4. Commit : `Add data contract with sample windows`.
2. **`src/preprocess.py`** (fonctions importables, utilisées aussi par la démo) :
   - Charger un fichier de poses. Une frame compte comme « sans point » si `person == 0` ; un keypoint est manquant si sa confiance `kp*_c < 0.3`.
   - Interpoler linéairement dans le temps, séquence par séquence, les trous de **15 frames au plus** (0,5 s). Les trous plus longs restent vides : ils sont fréquents quand la personne est allongée (35 % des frames sans squelette) et on ne les invente pas. `valid.npy` marque les vraies détections.
   - Normaliser : centre = milieu des hanches (kp11, kp12) de la frame ; échelle = **longueur du torse médiane de la séquence** (distance milieu des épaules (kp5, kp6) − milieu des hanches). Simplification à signaler : l'échelle utilise toute la séquence, donc pas du temps réel pur.
   - Découper en fenêtres de 30 frames, pas de 5, par séquence, sans traverser deux séquences. Calculer `y` (règle 7.0), `valid_frac` (part de frames réellement détectées), `fold`.
   - Écrire `windows_meta.parquet`, `X_seq.npy`, `valid.npy`, `folds.csv`. Les valeurs vides de `X_seq` sont mises à 0 (le masque `valid` les signale).
   - Commit : `Add preprocessing and sliding windows`.
3. **`src/features.py`** : signaux par frame, puis agrégats par fenêtre. Repère image : y vers le bas.
   - `torso_angle` : angle entre le torse (hanches → épaules) et la verticale ; `np.degrees(np.arctan2(abs(dx), -dy))` ; 0 debout, 90 couché.
   - `hip_vy` : vitesse verticale des hanches en longueurs de torse par seconde, positive vers le bas ; calculée sur les positions brutes en pixels divisées par l'échelle, lissée sur 3 frames. `hip_ay` : sa dérivée (accélération).
   - `bbox_ratio` : hauteur / largeur de la boîte YOLO (`by2-by1`, `bx2-bx1`).
   - `head_height` : (y des hanches − y de la tête) / échelle, positif quand la tête est au-dessus des hanches ; tête = moyenne des keypoints 0 à 4 disponibles.
   - Pour chaque signal, par fenêtre : `mean, min, max, std, delta` (dernière valeur − première), nommés `<signal>_<agg>` (ex. `torso_angle_max`). Ajouter `valid_frac`.
   - Les valeurs NaN (fenêtre quasi sans détection) restent NaN : chaque modèle les impute avec la **médiane du train**.
   - Écrire `features.parquet`. Commit : `Add per-window human features`.
4. **`src/metrics.py`** et **`src/evaluate.py`** : implémentent exactement les métriques de 7.0. `python src/evaluate.py results/<algo>_oof.parquet --tau 0.5` écrit `results/<algo>_metrics.csv` (une ligne par métrique). Teste-les sur le modèle naïf et sur des prédictions parfaites (tout doit sortir à 1). Commit : `Add shared evaluation metrics`.
5. **Modèle naïf** `src/run_naive.py` : prédit toujours `pas chute`. Écrit `results/naive_oof.parquet` et `models/naive.joblib`. Commit : `Add naive baseline`.
6. **Vérifications avant de dire « fini »** : nombre de fenêtres par classe et par fold (afficher le tableau) ; aucune séquence dans deux folds ; chaque fold contient des chutes ; l'ordre des lignes est cohérent entre les 4 fichiers ; la règle de label est écrite dans la docstring de `preprocess.py`.
7. **Ensuite** (si le temps le permet) : généraliser l'extraction de poses à n'importe quelle vidéo (option « image complète » au lieu de la moitié droite, 1280×720 pour GMDCSA-24 et pour les vidéos de l'équipe), puis `src/demo.py` (vidéo → poses → fenêtres → features → `predict_proba` de chaque algo → tableau de classes au fil du temps). Pour GMDCSA-24 : `make data-gmdcsa`, décompresser, extraire les poses, convertir les intervalles de `Classes` en classes par frame.

### 7.2 P2 — Règles puis régression (branche `p2-rules-logreg`)

Lis d'abord 7.0. Entrées : `processed/features.parquet` et `processed/windows_meta.parquet`.

1. **Règles à seuils (`src/run_rules.py`).** Déterministe, sans apprentissage au sens ML. C'est le repère de comparaison qui imite la téléassistance actuelle.
   - `chute` si, dans la fenêtre : vitesse descendante des hanches `hip_vy_max > s1` **et** angle du torse `torso_angle_max > s2`.
   - `allongé` si : `torso_angle_mean > s3` **et** `head_height_mean < s4` **et** `hip_vy_std` très faible (la personne ne bouge plus).
   - sinon `pas chute`.
   - **Les seuils `s1, s2, s3, s4` sont calés sur le train uniquement** : recherche sur grille, dans chaque fold, qui maximise le F2 de la classe `chute` (le rappel compte plus que la précision). Écris les seuils retenus dans `results/rules_thresholds.csv`.
   - Probabilités 0 ou 1.
2. **Régression logistique (`src/run_logreg.py`).** `Pipeline(SimpleImputer(median) → StandardScaler → LogisticRegression(multi_class multinomial, class_weight='balanced'))`. Choisis le paramètre `C` (grille `0.01, 0.1, 1, 10`) par découpage interne **par séquence** sur le train. Les features sont les colonnes de `features.parquet`.
3. Pour la régression : sauvegarde le tableau des coefficients (`results/logreg_coefficients.csv`) et écris 3 lignes d'interprétation (quelles features poussent vers `chute` ?). C'est vu en cours (L3).
4. Pour chaque algo : produis `results/<algo>_oof.parquet` et `models/<algo>.joblib` selon 7.0, lance `python src/evaluate.py ...`, **commite dès que l'algo tourne** (`Add threshold rules baseline`, `Add logistic regression`), puis pousse.
5. **Pièges :** régler les seuils ou `C` sur le fold de test ; oublier que l'imputation et la standardisation se calculent sur le train seulement ; ne pas fixer `random_state=42`.

### 7.3 P3 — Arbre puis boosting (branche `p3-trees`)

Lis d'abord 7.0. Mêmes entrées que P2.

1. **Arbre de décision (`src/run_cart.py`).** `Pipeline(SimpleImputer(median) → DecisionTreeClassifier(class_weight='balanced', random_state=42))`. Choisis `max_depth` dans `2..6` et `min_samples_leaf` dans `5, 10, 20` par découpage interne par séquence. Exporte l'arbre final lisible : `results/cart_tree.png` (`sklearn.tree.plot_tree`) et `results/cart_rules.txt` (`export_text`). L'intérêt de l'arbre est qu'il se lit comme des règles ; mets-le en face des règles de P2 dans les slides. Vu en cours (L5).
2. **Gradient boosting (`src/run_boosting.py`).** `GradientBoostingClassifier` de scikit-learn (pas LightGBM, qui demande `brew install libomp`), avec imputation médiane avant. Petite grille : `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {2, 3}, `learning_rate` ∈ {0.05, 0.1}, choisie par découpage interne par séquence. Vu en cours (L6).
3. **SHAP** : sur le modèle final entraîné sur toutes les données, `shap.TreeExplainer`, un graphique de résumé pour la classe `chute` (`results/shap_summary.png`) et l'importance moyenne par feature (`results/shap_importance.csv`). Écris 3 lignes : quelles features comptent, est-ce cohérent avec la physique d'une chute ?
4. Produis les sorties selon 7.0 (`results/cart_oof.parquet`, `results/boosting_oof.parquet`, `models/*.joblib`, `predict_proba`), lance `evaluate.py`, **commite dès que chaque algo tourne** (`Add decision tree`, `Add gradient boosting with SHAP`), puis pousse.
5. **Pièges :** les arbres sur-apprennent avec 30 chutes : un score trop beau sur le train n'a pas de valeur, regarde le test. Si le boosting a une AUC proche de 1, cherche une fuite avant de le croire. Ne pas régler sur le fold de test.

### 7.4 P4 — Deep learning (branche `p4-nn`)

Lis d'abord 7.0. Entrées : `processed/X_seq.npy` `(N, 30, 17, 2)`, `processed/valid.npy`, `processed/windows_meta.parquet`.

1. **Modèle (`src/run_nn.py`, PyTorch).** Deux variantes simples, à comparer sur le découpage interne :
   - MLP : aplatir à `1020` valeurs, `Linear(1020,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Linear(64,3)`. Vu en cours (L8-L9).
   - 1D-CNN : voir l'entrée comme `(34 canaux, 30 pas de temps)`, deux couches `Conv1d(k=5)` + `ReLU`, pooling global, `Linear(…,3)`.
   - Garde la meilleure dans `models/nn.pt`. Pas de LSTM (hors cours).
2. **Entraînement :** perte `CrossEntropyLoss` pondérée par l'inverse de la fréquence des classes (les chutes sont rares) ; `Adam` `lr=1e-3` ; 30 à 50 époques ; **arrêt anticipé sur un ensemble de validation tiré par séquence à l'intérieur du train** ; `torch.manual_seed(42)` et seed numpy partout. Tourne sur CPU ou MPS en quelques minutes.
3. **Entrées :** les valeurs vides ont été mises à 0 par P1 ; ajoute `valid` comme canal supplémentaire ou multiplie par le masque. Pas besoin d'autre normalisation (déjà centré et à l'échelle du torse), mais vérifie les ordres de grandeur.
4. **Augmentation (optionnelle)** : retournement horizontal (`x → −x`, avec échange des keypoints gauche/droite), petit bruit gaussien. Uniquement sur le train.
5. Produis `results/nn_oof.parquet`, `models/nn.pt` et `predict_proba` selon 7.0. Mesure et note le nombre de paramètres et la latence par fenêtre. Lance `evaluate.py`, **commite dès que l'algo tourne** (`Add neural network on raw keypoints`), puis pousse.
6. **Pièges :** avec 30 chutes, un réseau de neurones peut très bien faire moins bien que le boosting. C'est un résultat valable : ne force pas les chiffres, rapporte-le honnêtement. Ne jamais régler sur le fold de test. Fixer les seeds.

### 7.5 P5 — Comparaison finale (tous ensemble, branche `p5-compare`)

À faire quand tous les algos ont poussé leurs `results/<algo>_oof.parquet`.

1. **`src/compare.py`** : lit tous les `results/*_oof.parquet`, recalcule toutes les métriques avec `src/metrics.py` (pas celles des auteurs), et écrit `results/comparison.csv` et `results/comparison.md`. Une ligne par algo : rappel de chute (événements), délai moyen, fausses alertes (nombre et par heure), PR-AUC de `chute`, F1 par classe, latence, taille. Moyenne ± écart-type sur les 5 folds.
2. **Courbes** : `results/pr_curves.png` (précision/rappel de `chute`, tous les algos sur le même graphique) et `results/confusion_matrices.png`.
3. **Seuil `tau`** : balayer `tau` pour chaque algo ; choisir le seuil à partir du **coût** (une fausse alerte = une sortie inutile ; une chute ratée = du temps au sol), avec les paramètres de `config/impact.yaml`. Aucun chiffre inventé : chaque paramètre a une source ou est marqué « hypothèse », avec une analyse de sensibilité. Le seuil se choisit sur les folds de validation, pas sur ceux de test (voir L13 : un seuil choisi sur la validation a donné des résultats très différents en test).
4. **Analyse d'erreurs** : lister les séquences ADL qui déclenchent des fausses alertes (s'asseoir vite, se pencher, s'allonger) et les chutes ratées, avec 2 ou 3 images annotées pour les slides.
5. **Généralisation** : appliquer les modèles finaux (`models/*`) à GMDCSA-24 et aux vidéos de l'équipe (`data/own/`) via `src/demo.py`, regarder ce qui sort en classification pour chaque algo, et comparer avec URFD. C'est le test ultime (section 10). Dire clairement que la performance baisse quand on change de pièce, de caméra et de personnes.
6. Conclusion honnête : le modèle retenu et pourquoi, et ce qu'on ne peut pas affirmer avec 70 séquences et une seule configuration de caméra.

### 7.6 Autres livrables (proposition de répartition, à confirmer)

- `src/demo.py` et le test ultime : P1, une fois ses fondations finies.
- README d'une page, en anglais, pour un data scientist : P2.
- `config/impact.yaml` + `src/impact.py` : P3.
- Vidéo de 2 minutes : P4.
- Slides (5 à 10) et relecture finale : P5, avec tout le monde.

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

- Règle du label d'une fenêtre : proposée en 7.0 (classe de la dernière frame). P1 peut la changer en prévenant l'équipe.
- Les frames « allongé » des ADL (lit, canapé) comptent comme `allongé` : à confirmer, car ce n'est pas une conséquence de chute.
- CAUCAFall : on ne le télécharge pas (8,3 Go) sauf accord de l'équipe.
- Répartition de 7.6 (démo, README, impact, vidéo, slides) à confirmer.
- MP4 (320×240) ou PNG (640×480) : le test montre un gain faible, on reste sur MP4 sauf mauvaise surprise.
