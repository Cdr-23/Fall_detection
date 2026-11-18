# Plan initial du projet (archive)

> Instructions détaillées données à P1-P5 au début du projet. **Tout est terminé** ;
> l'état actuel et la suite sont dans la section 0 de `CLAUDE.md`. Conservé pour l'historique des choix.

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
7. **Ensuite** (si le temps le permet) : généraliser l'extraction de poses à n'importe quelle vidéo (option « image complète » au lieu de la moitié droite, 1280×720 pour GMDCSA-24 et pour les vidéos de l'équipe), puis `src/demo.py` (vidéo → poses → fenêtres → features → `predict_proba` de chaque algo → tableau de classes au fil du temps). Pour GMDCSA-24 : `python src/download.py --gmdcsa`, décompresser, extraire les poses, convertir les intervalles de `Classes` en classes par frame.

#### 7.1 bis — Livré par P1 (mardi 17 nov., branche `p1-foundations`)

**Ce qui est prêt** (tout tourne sous Windows et Mac, avec `python src/...` depuis la racine) :

| Fichier | Rôle |
|---|---|
| `processed/` | contrat 7.0 complet, 70 séquences, **2 019 fenêtres** : 1 373 `pas chute`, **151 `chute`**, 495 `allongé`. En plus : `sequences.csv` (`seq, kind, fold, n_frames, fall_start, fall_end`), utilisé par les métriques d'événements |
| `src/common.py` | `load_data()` → `meta, feats, X_seq, valid` ; `oof_frame(meta, proba)` → table OOF au bon format ; `save_oof(algo, oof)` ; `FEATURES` (les 26 colonnes) |
| `src/run_naive.py` | modèle de référence, à copier comme squelette (boucle sur les folds, `predict_proba`, sauvegarde) |
| `src/evaluate.py` | `python src/evaluate.py results/<algo>_oof.parquet` : vérifie le format, écrit `results/<algo>_metrics.csv` (valeur globale, moyenne ± écart-type sur les folds, chaque fold), mesure la latence de `predict_proba` et la taille du modèle |
| `tests/test_metrics.py` | `python tests/test_metrics.py` : prédictions parfaites → tout à 1, naïf → rappel 0, cas synthétiques, cohérence de `processed/` |
| `src/external.py` | GMDCSA-24 (et `data/own/`) → `poses/gmdcsa24/` et `processed/gmdcsa24/` au même format (`fold = -1`) |
| `src/demo.py` | `python src/demo.py video.mp4` ou `--poses poses/gmdcsa24/<seq>.parquet` : classe et `p1` de chaque algo au fil du temps (CSV + frise PNG dans `results/demo/`) |

**Conventions à connaître :**
- `predict_proba(feats, X_seq)` reçoit les lignes de `features.parquet` **avec** `window_id` : sélectionner `feats[FEATURES]`. La démo et `evaluate.py` l'appellent ainsi. Charger le modèle une seule fois (variable globale), comme dans `run_naive.py`.
- 44 fenêtres n'ont aucune feature (personne non détectée sur toute la fenêtre) : imputer avec la médiane du train.
- `.gitignore` laisse passer `models/*.pt` (pour `models/nn.pt`) ; `results/demo/` est ignoré (régénérable).

**Constats à reprendre dans les slides :**
- **16 chutes sur 30 commencent avant la frame 30** (la première seconde de la vidéo). La première fenêtre ne se termine qu'à la frame 30 : ces chutes ont moins de 6 fenêtres `chute` (151 au lieu de 180), peu de contexte « avant », et un délai minimal non nul. Même un modèle parfait a un délai moyen de **0,22 s**.
- Fausses alertes par heure : la durée d'ADL surveillée = nombre de fenêtres ADL × 5 / 30 s = **0,07 h** (les 30 premières frames de chaque séquence et les frames sans label ne sont jamais évaluées). Toujours donner le nombre brut.
- Modèle naïf : ROC-AUC 0,50, PR-AUC 0,075 (= part des fenêtres `chute`), rappel d'événements 0. C'est le plancher.
- Les poses ont été extraites sur Mac (MPS) ; refaire fall-01 sous Windows (CPU) donne 99 % de frames identiques en détection, à quelques pixels près. On garde les poses du repo.
- GMDCSA-24 : règle de label dans la docstring de `src/external.py` (« Falling » = `chute` pendant 1 s puis `allongé`, car l'intervalle court jusqu'à la fin du clip ; « Sleeping » = `allongé`). 29 clips sur 160 sont à 15 fps : les poses sont rééchantillonnées à 30 fps par interpolation.
- GMDCSA-24 est prêt (poses et fenêtres suivies par git, pas besoin de télécharger) : `poses/gmdcsa24/` (160 clips, personne détectée sur 95 % des frames) et `processed/gmdcsa24/` (**6 046 fenêtres** : 3 948 `pas chute`, 434 `chute`, 1 664 `allongé` ; 78 chutes avec un début connu). Le CSV source a des fautes : `s4-fall-15` n'a pas d'intervalle « Falling » (pas d'événement de chute, on ne le devine pas), un intervalle inversé est écarté (`s2-adl-06`), les autres sont corrigées par le parseur.
- Pour P5, appliquer un modèle final à GMDCSA-24 : `meta, feats, X, _ = load_data("processed/gmdcsa24")`, `proba = predict_proba(feats, X)`, `oof_frame(meta, proba)` → parquet, puis `python src/evaluate.py <fichier> --meta processed/gmdcsa24/windows_meta.parquet --sequences processed/gmdcsa24/sequences.csv`. Vidéos de l'équipe : `python src/external.py own` (avec `data/own/labels.csv`), même chose avec `processed/own`.

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

### 7.6 Autres livrables (ancienne proposition, remplacée par la section 0.3)

- `src/demo.py` et le test ultime : P1, une fois ses fondations finies.
- README d'une page, en anglais, pour un data scientist : P2.
- `config/impact.yaml` + `src/impact.py` : P3.
- Vidéo de 2 minutes : P4.
- Slides (5 à 10) et relecture finale : P5, avec tout le monde.

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

## 11. Livrables finaux (répartition : voir section 0.3)

- README d'une page pour un data scientist, code relançable avec un seul `make`.
- 5 à 10 slides adressées au décideur (opérateur de téléassistance).
- Vidéo de 2 minutes.
- Modèle d'impact (`config/impact.yaml`) : aucun chiffre inventé, chaque paramètre est sourcé ou marqué « hypothèse », avec analyse de sensibilité.

