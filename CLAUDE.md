# CLAUDE.md — Détection de chutes par squelette 2D (15.071 Projet 1)

> Brouillon de travail. Rendu : vendredi 20 novembre 2026.
> Langue de travail : français. Code, commentaires et README : anglais.

## 0. À LIRE EN PREMIER : état du projet et ce qu'il reste à faire (mis à jour le 18/11)

**Claude : si quelqu'un de l'équipe te demande « qu'est-ce que je dois faire ? », réponds à partir de cette section.** Les sections 1 à 9 plus bas donnent le contexte, les données, le contrat commun et les règles ; le plan détaillé d'origine (P1 à P5, terminé) est archivé dans `docs/plan_initial.md`. Les résultats détaillés et leur lecture sont dans `results/p5_notes.md` (à lire pour tout ce qui touche aux chiffres).

### 0.1 Ce qui est fait (tout est sur `main`)

- **Pipeline commun (P1) :** vidéo → squelette YOLOv8n-pose → fenêtres de 30 images (1 s) → 26 features + squelette brut. 3 classes : `pas chute` / `chute` / `allongé`. Validation croisée 5 folds par séquence sur URFD (70 vidéos).
- **6 modèles (P2-P4) :** naïf, règles à seuils, régression logistique, arbre, **gradient boosting**, réseau de neurones.
- **Comparaison (P5) :** `results/comparison.md`, `results/p5_notes.md`, figures dans `results/`.
- **Tests externes, jamais utilisés pour entraîner :** GMDCSA-24 (3 maisons) et Le2i (2 maisons).
- **Modèle d'impact :** `config/impact.yaml` (chaque paramètre sourcé ou marqué « hypothèse ») → `python src/impact.py` → `results/impact.md`, `impact_break_even.png`, `impact_tornado.png`.

### 0.2 Modèle retenu et chiffres clés (à reprendre tels quels dans les slides)

**Gradient boosting + confirmation : l'alerte ne part que si la personne est ensuite vue allongée (~0,5 s).** Choisi sur la validation croisée d'URFD (la bonne méthode : on ne choisit jamais un modèle sur un jeu de test).

| Boosting, sans → avec confirmation | URFD (validation croisée) | GMDCSA-24 (externe) | Le2i (externe) |
|---|---|---|---|
| Chutes détectées | 29/30 → 27/30 | 53/78 → 52/78 | 17/37 → 22/37 |
| Fourchette à 95 % (avec confirmation) | 74-97 % | 56-76 % | 43-74 % |
| Fausses alertes (vidéos d'activités normales) | 15 → 7 | 33 → 9 | 7 → 2 |
| Délai moyen de l'alerte | 0,4 s → 1,4 s | 0,3 s → 1,4 s | 0,2 s → 0,9 s |

Benchmarks : le modèle naïf détecte 0/30 chutes ; les règles à seuils (≈ « business rule ») 28/30 mais 30 fausses alertes et aucun classement (PR-AUC 0,12 contre 0,67 pour le boosting). Tous les modèles tournent en temps réel (~1 ms par fenêtre au plus).

**Les 6 messages à faire passer :**
1. **Le problème :** chez les plus de 90 ans, 60 % chutent dans l'année et 30 % des chuteurs restent ≥ 1 h au sol ; le médaillon d'alarme n'est pas utilisé dans la plupart de ces cas (Fleming & Brayne, BMJ 2008, https://doi.org/10.1136/bmj.a2227). 174 824 hospitalisations et 20 148 décès liés à une chute chez les 65+ en France en 2024 (Santé publique France, bulletin du 12/03/2026).
2. **La solution :** une caméra qui ne garde que le squelette (vie privée), un modèle simple et rapide, et une alerte en deux temps (détection puis confirmation au sol).
3. **Ça marche, avec des limites honnêtes :** 27/30 chutes sur URFD ; la performance baisse dans d'autres maisons (52/78 et 22/37), surtout avec une caméra placée haut.
4. **Principale erreur :** se pencher vite pour ramasser un objet ressemble à une chute sur 1 s de squelette (`results/errors/*.png`).
5. **Est-ce que ça compte (impact) :** pour 1 000 abonnés, ~58 longs séjours au sol (≥ 1 h) évités par an : c'est le bénéfice santé. En euros, **tout dépend du taux de fausses alertes en vie réelle**, qu'on ne peut pas mesurer : +291 € par personne et par an dans le scénario optimiste (1 fausse alerte tous les 100 jours), −108 € au centre (1 tous les 20 jours), −1 802 € au pessimiste (1 tous les 5 jours). **Rentable sous ~1 fausse alerte par personne tous les 48 jours** (`results/impact.md`, `results/impact_break_even.png`). Les euros sont un minimum : décès, entrées en EHPAD et qualité de vie ne sont pas comptés. D'où la recommandation : confirmation longue (10-30 s au sol), appel de vérification avant toute sortie, et un pilote pour mesurer le vrai taux de fausses alertes (critère clair de go / no-go).
6. **Recommandation :** un pilote chez un opérateur de téléassistance, pour mesurer le vrai taux de fausses alertes. C'est la donnée qui manque.

**Ce qu'on ne doit PAS dire :** un taux de fausses alertes par heure en vie réelle (nos vidéos d'activités ne font que quelques minutes) ; un taux de détection précis (27/30 = entre 74 et 97 %) ; que c'est déjà du temps réel strict (section 9) ; que ça marche partout (3 datasets, peu de personnes) ; que les euros de l'impact sont certains (plusieurs paramètres sont des hypothèses, marqués comme tels).

### 0.3 Qui fait quoi d'ici vendredi 20/11

| Qui | Tâche |
|---|---|
| Aristote | README d'une page (anglais), modèle d'impact (v1 faite), relecture des chiffres |
| **Alain et Jules** | **Vidéos de test (0.4), slides (0.5), vidéo de 2 min (0.6)** |

### 0.4 Alain et Jules : tourner et traiter les vidéos de test (« test ultime »)

**But :** voir ce que sort chaque modèle sur nos propres vidéos (seul test vraiment à l'aveugle), et mesurer la confirmation longue (10-30 s au sol), impossible avec les datasets publics.

**Tournage :**
- Caméra fixe (téléphone sur un support), **à l'horizontale**, à ~1-1,5 m de haut, de côté ou en diagonale, **corps entier dans le cadre**, pièce bien éclairée. (URFD, sur lequel le modèle a appris, est filmé ainsi. 1 ou 2 clips avec la caméra en hauteur montreront la limite.)
- **Chutes** (sur un matelas, en sécurité) : 5 à 10 clips, en avant, en arrière, sur le côté, depuis une chaise. **Rester au sol au moins 30 s** après chaque chute, sans se relever.
- **Activités normales** : 5 à 10 clips d'**au moins 1-2 min en continu** : marcher, **se pencher pour ramasser un objet** (plusieurs fois), s'asseoir vite, s'allonger sur un lit ou un canapé.
- Noms de fichiers **obligatoires** : `fall-01.mp4`, `fall-02.mp4`, ..., `adl-01.mp4`, ... (le mot `fall` ou `adl` dit au code si c'est une chute).
- Mettre les vidéos dans `data/own/`. **Elles ne vont pas sur GitHub** (`data/` est ignoré : visages de l'équipe).

**Étiquettes :** un fichier `data/own/labels.csv` avec les colonnes `file,classes`, temps en secondes depuis le début du clip, par exemple :
```
file,classes
fall-01.mp4,"Walking[0 to 4.2]; Falling[4.2 to 40]"
adl-01.mp4,"Walking[0 to 20]; Bending[20 to 23]; Walking[23 to 60]; Lying[60 to 90]"
```
Règle du code : un intervalle `Falling[a to b]` = `chute` pendant 1 s à partir de `a`, puis `allongé` jusqu'à `b` ; `Lying`/`Sleeping` = `allongé` ; tout le reste (`Walking`, `Bending`, `Sitting`...) = `pas chute`.

**Commandes** (environnement activé, voir section 3) :
```bash
python src/external.py own          # squelettes + fenêtres -> poses/own, processed/own
python src/confirm_long.py          # confirmation 0,5 à 30 s au sol -> results/confirm_long.csv (lignes dataset = own)
python src/demo.py data/own/fall-01.mp4 --labels "Walking[0 to 4.2]; Falling[4.2 to 40]"   # classe prédite au fil du temps, chaque modèle
```
`demo.py` écrit `results/demo/<nom>.csv` et `.png` (frise : vérité vs prédiction de chaque modèle). `results/demo/` est ignoré par git : **copier dans `results/own/` les 2-3 frises utiles aux slides** et les commiter.

**Règles :** ne rien régler (seuil, modèle) sur ces vidéos, sinon ce n'est plus un test ; ne pas relancer `src/extract_pose.py` sur URFD ; commiter `poses/own`, `processed/own`, `results/own/`, `results/confirm_long.csv`, sur une branche `p6-videos-slides`, sans mention de Claude dans les commits.

### 0.5 Alain et Jules : les slides (5 à 10, adressées à la BSPP ou à un opérateur de téléassistance)

La consigne du cours (document « Team project description », 15.071) : problème, solution IA, impact, recommandations ; s'adresser au décideur, pas à un data scientist ; « un modèle simple qui règle un vrai problème vaut mieux qu'un système complexe ». Proposition :

| # | Slide | Contenu et source |
|---|---|---|
| 1 | Le problème | chiffres SpF 2024, long séjour au sol et médaillon non utilisé (Fleming & Brayne) (message 1) |
| 2 | La solution | caméra → squelette (vie privée) → modèle → alerte en 2 temps (schéma) |
| 3 | Les données | URFD (entraînement, 70 vidéos), GMDCSA-24 et Le2i (tests externes), nos vidéos (test final) |
| 4 | Les modèles comparés | 6 modèles, du naïf au réseau ; boosting retenu ; `results/pr_curves.png` |
| 5 | Est-ce que ça marche | tableau 0.2 ; une image d'erreur (`results/errors/boosting_adl-05.png`) |
| 6 | Test sur nos vidéos | 1-2 frises `results/own/*.png` (0.4) |
| 7 | Est-ce que ça compte | ~58 longs séjours au sol évités ; tableau des 3 scénarios (`results/impact.md`) ; `results/impact_break_even.png` (message 5) |
| 8 | Recommandation et limites | pilote, confirmation longue + appel ; ce qu'on ne peut pas affirmer |

### 0.6 Vidéo de 2 minutes (pitch au décideur)

~20 s le problème (une chute, personne pour aider) → ~30 s la solution (squelette, alerte en 2 temps) → ~40 s démo sur une de nos vidéos (frise `demo.py`) → ~20 s les chiffres (27/30 chutes, limites) → ~10 s la recommandation (pilote).

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

**Prérequis (Mac, Windows ou Linux) :** `git`, **Python 3.12** ([python.org](https://www.python.org/downloads/), ou `brew install python@3.12` sur Mac) et un accès en écriture au repo GitHub (à demander au propriétaire). Rien d'autre à télécharger à la main. Les commandes ci-dessous n'utilisent que Python : elles marchent partout. Le `Makefile` n'est qu'un raccourci pour Mac/Linux (il demande `make` et `uv`).

```bash
git clone https://github.com/Cdr-23/Fall_detection
cd Fall_detection
```

Crée et active l'environnement, puis installe (une seule fois) :

| | Mac / Linux | Windows (PowerShell) |
|---|---|---|
| Créer | `python3.12 -m venv .venv` | `py -3.12 -m venv .venv` |
| Activer | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| Installer | `pip install -r requirements.txt` | `pip install -r requirements.txt` |

Sous Windows, si l'activation est bloquée : `Set-ExecutionPolicy -Scope Process RemoteSigned`. Les modèles tournent sur CPU (c'est suffisant) ; une carte NVIDIA est utilisée automatiquement si PyTorch la voit. Vérifié sur Mac avec `pip` ; **pas testé sous Windows** : si une étape échoue, le dire à l'équipe plutôt que de contourner.

**Les poses sont déjà dans le repo** (`poses/<séquence>.parquet`, un fichier par séquence URFD, avec les labels). P2, P3 et P4 peuvent donc coder leur algo tout de suite, sans télécharger les vidéos ni relancer YOLO.

| Commande (l'environnement est activé) | Ce que ça fait | Qui |
|---|---|---|
| `pip install -r requirements.txt` | installe les bibliothèques | tout le monde |
| `python src/download.py` | télécharge URFD (115 Mo) et vérifie les 70 vidéos | seulement pour la démo, les visuels ou pour refaire les poses |
| `python src/extract_pose.py` | extrait les poses avec YOLOv8n-pose (~7 min sur Mac) ; saute les séquences déjà faites | P1, si on change de modèle ou de réglage |
| `python src/download.py --gmdcsa` | télécharge aussi GMDCSA-24 (1,1 Go), test externe | P1 seulement |

(Sur Mac/Linux, `make setup`, `make data`, `make pose` et `make data-gmdcsa` font la même chose.)

Ne pas relancer l'extraction de poses sans nécessité : le résultat est partagé, et deux versions des poses rendraient les comparaisons incohérentes.

**Format des poses** (`poses/<seq>.parquet`, une ligne par frame) : `seq, frame, label, person, kp0_x, kp0_y, kp0_c ... kp16_x, kp16_y, kp16_c, bx1, by1, bx2, by2, img_w, img_h`.
- `frame` est numéroté à partir de 1, comme dans les CSV de labels d'URFD.
- `label` (colonne URFD) : -1 pas allongé, 1 allongé, 0 pose temporaire (« en train de tomber » dans les séquences `fall-*`, mais aussi s'asseoir ou se pencher dans les `adl-*`). Le passage aux 3 classes est défini en 7.0. NaN quand la frame n'est pas dans le CSV d'origine (certaines frames des ADL n'ont pas de label : à ignorer ou à documenter).
- `person` vaut 0 quand YOLO ne trouve personne ; les keypoints sont alors vides. C'est fréquent quand la personne est allongée (~30 % des frames).
- Points COCO (0 nez, 5-6 épaules, 11-12 hanches, 13-14 genoux, 15-16 chevilles), en pixels de l'image RGB 320×240, avec leur confiance `_c`.

**Les données brutes (pour info)**
- URFD : 70 MP4 (640×240 : profondeur à gauche, **RGB à droite**) + 2 CSV de labels. Les frames des vidéos correspondent aux labels (vérifié : 11 936 frames, 0 écart). Licence CC BY-NC-SA 4.0 : Kwolek & Kepski, 2014, https://fenix.ur.edu.pl/mkepski/ds/uf.html
- GMDCSA-24 (`python src/download.py --gmdcsa`, télécharge seulement le zip ; ensuite `cd data/raw/gmdcsa24 && unzip -q GMDCSA24-v2.1.zip`) : 160 clips MP4 (79 chutes, 81 ADL), 4 sujets, 3 domiciles, licence CC BY 4.0, DOI 10.5281/zenodo.13354453. Vérifié : le zip est intact. Structure : `<dossier>/Subject N/Fall/NN.mp4`, `Subject N/ADL/NN.mp4`, plus `Subject N/Fall.csv` et `ADL.csv`. Les vidéos sont en **1280×720, ~30 fps, image RGB complète** (pas de profondeur : ne pas recadrer comme pour URFD). Les CSV ont les colonnes `File Name, Length (seconds), Time of Recording, Attire, Description, Classes` ; `Classes` donne des intervalles en secondes, par ex. `Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]` ou `Sitting[0 to 1]; Sleeping[2 to 8]`. Attention : « Falling » court jusqu'à la fin du clip (chute + sol), donc c'est plus grossier que les labels d'URFD. `src/extract_pose.py` ne gère pas ce format (il lit les MP4 d'URFD et garde la moitié droite) : il faut le généraliser avant de s'en servir.
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
| Test externe 2 | **Le2i FDD** (Home_01 + Home_02 seulement) | 60 clips (37 chutes, 23 ADL), 2 domiciles, 24 fps | `data/raw/le2i/` (2,8 Go, [Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia), compte requis) ; puis `python src/le2i_convert.py` et `python src/external.py le2i` |
| Test final | **Vidéos filmées par l'équipe** | chutes et activités dans un environnement clair | `data/own/` (voir format ci-dessous) |
| En attente | CAUCAFall | 8,3 Go ; Mendeley en panne le 18/11 | non utilisé |

- `data/` est exclu de git (`.gitignore`). Les données brutes se téléchargent avec `python src/download.py` (voir section 3). Les poses extraites (`poses/`, 6 Mo) sont suivies par git.
- IKAR (AGH Cracovie, 278 vidéos de chutes, sans activités du quotidien) est écarté : on n'en a pas besoin, les vidéos de l'équipe le remplacent.
- **Personne n'a besoin de télécharger quoi que ce soit pour coder P2, P3, P4** : les poses sont dans le repo. `python src/download.py` (URFD, 115 Mo) ne sert qu'à la démo et aux visuels ; GMDCSA-24 (1,1 Go) ne sert qu'au test externe.
- Les labels n'ont pas le même format selon le dataset : une règle unique « fenêtre = chute ? » est à écrire et à documenter.
- Les datasets sont sous licence CC BY (-NC-SA pour URFD) : à citer dans le README.

**Vidéos de l'équipe (proposition de format) :** fichiers `data/own/fall-01.mp4`, `adl-01.mp4`, ... en ~30 fps, personne entière dans le cadre, caméra fixe, au moins une chute jouée sur tapis et des activités normales (s'asseoir, se pencher, s'allonger volontairement). Un fichier `data/own/labels.csv` avec les colonnes `file, classes`, où `classes` suit le style de GMDCSA-24 : `Falling[3.4 to 4.4]; Lying[4.4 to 10]`. Sans labels, on peut quand même regarder la sortie des algos, mais pas calculer de métriques. Ces vidéos ne servent qu'à tester.

**Pour tester la confirmation longue (10-30 s au sol), recommandée pour le produit :** après chaque chute, rester **au moins 30 s au sol** sans se relever, et filmer les activités normales **en continu pendant plusieurs minutes** (se pencher pour ramasser un objet, s'asseoir vite, s'allonger sur un lit ou un canapé). Les vidéos publiques s'arrêtent 2 à 4 s après la chute : seules nos vidéos peuvent valider ce réglage. Ensuite : `python src/external.py own` puis `python src/confirm_long.py`.

## 6. Pipeline commun (déjà fixé)

1. **Pose :** YOLOv8n-pose (déjà entraîné, utilisé tel quel) → 17 points par frame, mis en cache dans `poses/`.
2. **Préparation :** normalisation (centre = milieu des hanches, échelle = longueur du torse), interpolation des points manquants, fenêtres de 30 frames, pas de 5.
3. **Features :** angle du torse, vitesse et accélération verticales des hanches, ratio hauteur/largeur de la boîte, hauteur de la tête, statistiques par fenêtre.

Limite connue : YOLO détecte mal une personne allongée (une personne trouvée sur 65 % des frames allongées, contre 95 % des frames debout). À dire honnêtement dans les slides.

## 7. Les algorithmes et le contrat commun

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

(Les instructions détaillées de P1 à P5, toutes terminées, sont archivées dans [`docs/plan_initial.md`](docs/plan_initial.md).)

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
| `features.parquet` | `window_id` + les colonnes de features (liste dans `src/features.py`) |
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

## 9. Choix assumés (à savoir expliquer si on nous le demande)

- **Label d'une fenêtre = classe de sa dernière image** : le modèle dit ce qui se passe *maintenant*, ce qui marche en temps réel. Choix d'équipe, appliqué pareil à tous les modèles.
- **S'allonger volontairement (lit, canapé) = `allongé`** : ce n'est pas une chute, et l'alerte ne part que sur `chute` (puis confirmation). Limite : un mouvement brusque suivi d'un allongement volontaire peut encore déclencher une alerte.
- **Normalisation par la taille du torse sur toute la vidéo** (`src/preprocess.py`) : pas strictement temps réel ; en déploiement, il faudrait une estimation au fil de l'eau (résultats peut-être un peu optimistes).
- **Seuil d'alerte 0,5** : cohérent avec le modèle d'impact (une chute ratée ≈ 70 fausses alertes ; le seuil choisi sans le fold de test est 0,50-0,75 pour le boosting, mêmes fausses alertes).
- **CAUCAFall non utilisé** (Mendeley en panne le 18/11) ; vidéos URFD en MP4 320×240 (le PNG 640×480 n'apportait presque rien).
