"""Streamlit Dashboard d'Analyses Statistiques
------------------------------------------------
Fonctionnalités :
1. Télécharge un fichier JSON distant et charge les données dans un DataFrame.
2. Détection automatique des variables quantitatives / qualitatives.
3. Tests statistiques interactifs :
   - Test t de Student (2 groupes)
   - Corrélation de Pearson (2 variables quantitatives)
   - Khi2 d'indépendance (2 qualitatives)
   - Kruskal-Wallis (>=2 groupes, non paramétrique)
   - ANOVA à 1 facteur (>=2 groupes, paramétrique)
4. Visualisations adaptées (boxplots, scatter, heatmap, distribution, etc.).
5. Interprétation pédagogique des p-values avec niveau alpha configurable.
6. Gestion robuste des données manquantes et petits échantillons.

Exécution :
	streamlit run final.py

Prérequis (si besoin) :
	pip install streamlit pandas numpy seaborn matplotlib scipy requests

Hypothèses si schéma incertain :
 - Variables de type object => qualitatives.
 - Variables numériques (int/float) => quantitatives.
 - Si > 30 modalités pour une variable object on la considère peut‑être plutôt comme identifiant et on l'exclut des tests de groupe.
"""

from __future__ import annotations

import io
import json
import textwrap
from typing import Dict, List, Tuple

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tentative d'import de seaborn (optionnel pour déploiement allégé)
try:  # pragma: no cover - garde-fou environnement
	import seaborn as sns  # type: ignore
	HAS_SEABORN = True
except Exception:  # noqa: BLE001
	HAS_SEABORN = False

# Import SciPy (fortement recommandé). Si absent on désactive les tests.
try:  # pragma: no cover
	from scipy.stats import (  # type: ignore
		ttest_ind,
		pearsonr,
		chi2_contingency,
		kruskal,
		f_oneway,
	)
	HAS_SCIPY = True
except Exception:  # noqa: BLE001
	HAS_SCIPY = False
	# Définitions minimales pour éviter les plantages si SciPy manque
	def _scipy_missing(*_, **__):
		raise ImportError("SciPy n'est pas installé : installez 'scipy' pour activer les tests statistiques.")

	ttest_ind = pearsonr = chi2_contingency = kruskal = f_oneway = _scipy_missing  # type: ignore
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration Streamlit & Style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Statistique", layout="wide")
if 'sns' in globals() and HAS_SEABORN:
	try:
		sns.set_theme(style="whitegrid")
	except Exception:
		pass
CUSTOM_CSS = """
<style>
/* Global tweaks */
.block-container{padding-top:1rem; max-width:1300px;}
h1, h2, h3 {font-family: 'Segoe UI', sans-serif; font-weight:600;}
.stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
.stTabs [data-baseweb="tab"] {background: #f5f7fa; padding:10px 16px; border-radius:6px;}
.stTabs [aria-selected="true"] {background:#2563eb; color:#fff;}
div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e5e7eb; padding:8px 12px; border-radius:8px;}
.test-box{background:#ffffff;border:1px solid #e5e7eb;padding:1rem 1.1rem;margin-bottom:1rem;border-radius:10px;}
.sig-yes{color:#065f46;font-weight:600;}
.sig-no{color:#92400e;font-weight:600;}
.stAlert{padding:0.75rem 0.9rem;}
/* Hide fullscreen button on figures to keep UI minimal */
button[kind="header"] {display:none;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_URL = "https://edumail.fr/formations/realtimedata.json"
LOCAL_FALLBACK_PATH = "Data/real_time_data.json"  # Fallback local si URL indisponible
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Chargement des données (avec cache)
# ---------------------------------------------------------------------------
def _json_to_df(data) -> pd.DataFrame:
	"""Convertit un objet JSON en DataFrame avec heuristiques (list/dict)."""
	if isinstance(data, list):
		if all(isinstance(e, dict) for e in data):
			return pd.DataFrame(data)
		return pd.DataFrame({"valeur": data})
	if isinstance(data, dict):
		candidate = None
		max_len = -1
		for k, v in data.items():
			if isinstance(v, list) and len(v) > max_len and all(isinstance(e, dict) for e in v):
				candidate = v
				max_len = len(v)
		if candidate is not None:
			return pd.DataFrame(candidate)
		return pd.json_normalize(data)
	raise ValueError("Format JSON non supporté pour conversion en DataFrame")

@st.cache_data(show_spinner=True)
def load_json_from_url(url: str) -> pd.DataFrame:
	headers = {"User-Agent": "Mozilla/5.0 (compatible; StatDashboard/1.0)"}
	resp = requests.get(url, timeout=20, headers=headers)
	resp.raise_for_status()
	return _json_to_df(resp.json())

@st.cache_data(show_spinner=False)
def load_local_file(path: str) -> pd.DataFrame:
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	return _json_to_df(data)

def load_from_text(raw: str) -> pd.DataFrame:
	return _json_to_df(json.loads(raw))

def fetch_raw_json_from_url(url: str) -> dict:
	"""Télécharge et retourne le JSON brut (dict ou list)."""
	headers = {"User-Agent": "Mozilla/5.0 (compatible; StatDashboard/1.0)"}
	resp = requests.get(url, timeout=20, headers=headers)
	resp.raise_for_status()
	return resp.json()

def read_raw_local(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


# ---------------------------------------------------------------------------
# Détection des types de variables
# ---------------------------------------------------------------------------
def detect_variable_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	"""Retourne (quantitatives, qualitatives) selon heuristiques."""
	quantitative = df.select_dtypes(include=["number"]).columns.tolist()
	qualitative = df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()

	# Filtrer variables qualitatives suspectes (trop de modalités => id potentiels)
	filtered_qual = []
	for col in qualitative:
		nunique = df[col].nunique(dropna=True)
		if nunique > 0 and nunique <= 50:  # seuil empirique
			filtered_qual.append(col)
	return quantitative, filtered_qual


# ---------------------------------------------------------------------------
# Helpers d'interprétation
# ---------------------------------------------------------------------------
def interpret_pvalue(p: float, alpha: float) -> str:
	if np.isnan(p):
		return "P-value indisponible (données insuffisantes)."
	if p < alpha:
		return f"p={p:.3g} < α={alpha} -> Résultat significatif (on rejette H0)."
	return f"p={p:.3g} ≥ α={alpha} -> Non significatif (on ne rejette pas H0)."

def _format_significance(p: float, alpha: float) -> str:
	cls = 'sig-yes' if p < alpha else 'sig-no'
	lab = 'Significatif' if p < alpha else 'Non significatif'
	return f"<span class='{cls}'>{lab}</span> (p={p:.3g})"

def render_test_box(title: str, body_render_fn):
	with st.container():
		st.markdown(f"### {title}")
		body_render_fn()

def _metric_row(cols, items):
	for c, (label, value) in zip(cols, items):
		c.metric(label, value)


# ---------------------------------------------------------------------------
# Fonctions de tests statistiques (retour dict structuré)
# ---------------------------------------------------------------------------
def run_ttest(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
	groups = df[group_col].dropna().unique()
	if len(groups) < 2:
		return {"ok": False, "msg": "Moins de deux groupes."}
	g1 = df[df[group_col] == groups[0]][value_col].dropna()
	g2 = df[df[group_col] == groups[1]][value_col].dropna()
	if len(g1) < 2 or len(g2) < 2:
		return {"ok": False, "msg": "Échantillons trop petits."}
	stat, p = ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
	return {"ok": True, "stat": stat, "pvalue": p, "groups": (groups[0], groups[1])}


def run_pearson(df: pd.DataFrame, x: str, y: str) -> Dict:
	sub = df[[x, y]].dropna()
	if len(sub) < 3:
		return {"ok": False, "msg": "Données insuffisantes (<3)."}
	r, p = pearsonr(sub[x], sub[y])
	return {"ok": True, "stat": r, "pvalue": p, "n": len(sub)}


def run_chi2(df: pd.DataFrame, c1: str, c2: str) -> Dict:
	contingency = pd.crosstab(df[c1], df[c2])
	if contingency.size == 0 or contingency.shape[0] < 2 or contingency.shape[1] < 2:
		return {"ok": False, "msg": "Table de contingence insuffisante."}
	chi2, p, dof, expected = chi2_contingency(contingency)
	return {
		"ok": True,
		"stat": chi2,
		"pvalue": p,
		"dof": dof,
		"contingency": contingency,
		"expected": expected,
	}


def run_kruskal(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
	groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
	groups = [g for g in groups if len(g) > 0]
	if len(groups) < 2:
		return {"ok": False, "msg": "Moins de deux groupes valides."}
	stat, p = kruskal(*groups)
	return {"ok": True, "stat": stat, "pvalue": p, "k": len(groups)}


def run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict:
	groups = [g[value_col].dropna() for _, g in df.groupby(group_col)]
	groups = [g for g in groups if len(g) > 1]
	if len(groups) < 2:
		return {"ok": False, "msg": "Besoin d'au moins deux groupes avec >=2 valeurs."}
	stat, p = f_oneway(*groups)
	return {"ok": True, "stat": stat, "pvalue": p, "k": len(groups)}


# ---------------------------------------------------------------------------
# Visualisations utilitaires
# ---------------------------------------------------------------------------
def make_boxplot(df: pd.DataFrame, x: str, y: str):
	fig, ax = plt.subplots(figsize=(5, 3.5))
	if HAS_SEABORN:
		sns.boxplot(data=df, x=x, y=y, ax=ax)
		sns.stripplot(data=df, x=x, y=y, ax=ax, color="black", size=3, alpha=0.5)
	else:  # fallback matplotlib
		cats = list(df[x].dropna().unique())
		data = [df[df[x] == c][y].dropna().values for c in cats]
		ax.boxplot(data, labels=cats, patch_artist=True)
		for i, vals in enumerate(data, start=1):
			if len(vals):
				ax.scatter(np.random.normal(i, 0.04, len(vals)), vals, s=15, alpha=0.6, color='black')
	ax.set_title(f"Distribution de {y} selon {x}")
	return fig


def make_scatter(df: pd.DataFrame, x: str, y: str):
	fig, ax = plt.subplots(figsize=(5, 3.5))
	if HAS_SEABORN:
		sns.regplot(data=df, x=x, y=y, ax=ax, scatter_kws={"alpha": 0.7})
	else:
		ax.scatter(df[x], df[y], alpha=0.7)
		if df[x].notna().sum() > 1 and df[y].notna().sum() > 1:
			try:
				a, b = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
				xs = np.linspace(df[x].min(), df[x].max(), 100)
				ax.plot(xs, a*xs + b, color='red')
			except Exception:
				pass
	ax.set_title(f"Relation {x} vs {y}")
	return fig


def make_heatmap(contingency: pd.DataFrame):
	fig, ax = plt.subplots(figsize=(4 + 0.3 * contingency.shape[1], 3 + 0.3 * contingency.shape[0]))
	if HAS_SEABORN:
		sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", ax=ax)
	else:
		data = contingency.values
		im = ax.imshow(data, cmap='Blues')
		ax.set_xticks(range(data.shape[1]))
		ax.set_yticks(range(data.shape[0]))
		ax.set_xticklabels(contingency.columns)
		ax.set_yticklabels(contingency.index)
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				ax.text(j, i, int(data[i, j]), ha='center', va='center', color='black')
		fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	ax.set_title("Table de contingence")
	return fig


def make_distplot(df: pd.DataFrame, col: str):
	fig, ax = plt.subplots(figsize=(5, 3))
	series = df[col].dropna()
	if HAS_SEABORN:
		sns.histplot(series, kde=True, ax=ax)
	else:
		ax.hist(series, bins='auto', alpha=0.75, color='#2563eb')
	ax.set_title(f"Distribution de {col}")
	return fig

def make_regression(df: pd.DataFrame, x: str, y: str):
	"""Scatter + droite de régression (OLS simple)."""
	fig, ax = plt.subplots(figsize=(6,4))
	if HAS_SEABORN:
		try:
			sns.scatterplot(data=df, x=x, y=y, ax=ax, alpha=0.6, edgecolor=None)
		except Exception:
			ax.scatter(df[x], df[y], alpha=0.6)
	else:
		ax.scatter(df[x], df[y], alpha=0.6)
	if df[x].notna().sum() > 1 and df[y].notna().sum() > 1:
		try:
			a, b = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
			x_line = np.linspace(df[x].min(), df[x].max(), 100)
			y_line = a * x_line + b
			ax.plot(x_line, y_line, color='red', label=f"y = {a:.3f}x + {b:.3f}")
			ax.legend()
		except Exception:
			pass
	ax.set_title(f"Régression {y} ~ {x}")
	return fig

def make_side_by_side_hists(list_a, list_b, labels=("A","B")):
	fig, ax = plt.subplots(figsize=(5,3.5))
	ax.hist(list_a, alpha=0.5, label=labels[0])
	ax.hist(list_b, alpha=0.5, label=labels[1])
	ax.set_title("Distributions")
	ax.legend()
	return fig

def make_violin_two(list_a, list_b, labels=("A","B")):
	fig, ax = plt.subplots(figsize=(5,3.5))
	ax.violinplot([list_a, list_b], showmeans=True)
	ax.set_xticks([1,2], labels)
	ax.set_title("Violin plot")
	return fig

def make_course_format_bar(cont: pd.DataFrame):
	fig, ax = plt.subplots(figsize=(5,3.5))
	# cont index = modalité, colonnes = resultats
	stack_bottom = np.zeros(len(cont.index))
	for col in cont.columns:
		vals = cont[col].values
		ax.bar(cont.index, vals, bottom=stack_bottom, label=col)
		stack_bottom += vals
	ax.set_ylabel('Effectif')
	ax.set_title('Répartition par modalité')
	ax.legend()
	return fig


# ---------------------------------------------------------------------------
# Barre latérale : Paramètres globaux
# ---------------------------------------------------------------------------
with st.sidebar:
	st.header("Paramètres")
	alpha = st.number_input("Niveau de significativité α", min_value=0.001, max_value=0.2, value=DEFAULT_ALPHA, step=0.01, format="%.3f")
	st.markdown("---")
	st.subheader("Source des données")
	source_choice = st.radio(
		"Choisir la source",
		["URL distante", "Fichier local", "Upload", "Coller JSON"],
		index=0,
		help="Si l'URL renvoie 403, utilisez un fallback.")
	uploaded_file = None
	pasted_json = None
	if source_choice == "Upload":
		uploaded_file = st.file_uploader("Charger un fichier JSON", type="json")
	elif source_choice == "Coller JSON":
		pasted_json = st.text_area("Coller le contenu JSON", height=160)
	st.markdown("---")
	st.caption("Ce dashboard effectue plusieurs tests statistiques classiques.")


# ---------------------------------------------------------------------------
# Chargement & aperçu
# ---------------------------------------------------------------------------
st.title("📊 Dashboard Statistique Interactif")
st.markdown("Sélectionnez les variables pour lancer les tests. Changez la source si nécessaire.")

# Avertissements dépendances manquantes
if not HAS_SEABORN:
	st.warning("Seaborn non installé - utilisation de graphes matplotlib simplifiés. Ajoutez 'seaborn' dans requirements.txt pour des visuels enrichis.")
if not HAS_SCIPY:
	st.error("SciPy non installé - les tests statistiques sont désactivés. Ajoutez 'scipy' dans requirements.txt.")

# Chargement dynamique selon la source
df = pd.DataFrame()
load_error = None
raw_json = None
if source_choice == "URL distante":
	try:
		raw_json = fetch_raw_json_from_url(DATA_URL)
		# Utiliser de préférence l'entrée 'studyCorrelation' si présente
		if isinstance(raw_json, dict) and 'studyCorrelation' in raw_json:
			try:
				df = _json_to_df(raw_json['studyCorrelation'])
			except Exception:
				df = load_json_from_url(DATA_URL)
		else:
			df = load_json_from_url(DATA_URL)
	except Exception as e:
		load_error = f"Erreur téléchargement distant: {e}"
		try:
			raw_json = read_raw_local(LOCAL_FALLBACK_PATH)
			if isinstance(raw_json, dict) and 'studyCorrelation' in raw_json:
				df = _json_to_df(raw_json['studyCorrelation'])
			else:
				df = load_local_file(LOCAL_FALLBACK_PATH)
			st.warning(f"{load_error}\nFallback local utilisé: {LOCAL_FALLBACK_PATH}")
		except Exception as e2:
			load_error = f"{load_error}\nFallback local impossible: {e2}"
elif source_choice == "Fichier local":
	try:
		raw_json = read_raw_local(LOCAL_FALLBACK_PATH)
		if isinstance(raw_json, dict) and 'studyCorrelation' in raw_json:
			df = _json_to_df(raw_json['studyCorrelation'])
		else:
			df = load_local_file(LOCAL_FALLBACK_PATH)
	except Exception as e:
		load_error = f"Erreur lecture locale: {e}"
elif source_choice == "Upload" and uploaded_file is not None:
	try:
		content = uploaded_file.read().decode("utf-8")
		raw_json = json.loads(content)
		if isinstance(raw_json, dict) and 'studyCorrelation' in raw_json:
			df = _json_to_df(raw_json['studyCorrelation'])
		else:
			df = _json_to_df(raw_json)
	except Exception as e:
		load_error = f"Erreur parsing fichier uploadé: {e}"
elif source_choice == "Coller JSON" and pasted_json:
	try:
		raw_json = json.loads(pasted_json)
		if isinstance(raw_json, dict) and 'studyCorrelation' in raw_json:
			df = _json_to_df(raw_json['studyCorrelation'])
		else:
			df = _json_to_df(raw_json)
	except Exception as e:
		load_error = f"Erreur parsing JSON collé: {e}"

if load_error and df.empty:
	st.error(load_error)
	st.stop()
elif load_error and not df.empty:
	st.info("Une erreur est survenue mais un fallback a été chargé.")

if df.empty:
	st.warning("Le DataFrame est vide. Changez de source de données dans la barre latérale.")
	st.stop()

quant_vars, cat_vars = detect_variable_types(df)

# ---------------------------------------------------------------------------
# Enrichissement à partir du JSON complet (midtermScores, approachGains, etc.)
# ---------------------------------------------------------------------------
midterm_df = None
approach_df = None
course_format_df = None
completion_df = None

def _safe_get(d: dict, key: str, default=None):
	return d.get(key, default) if isinstance(d, dict) else default

if isinstance(raw_json, dict):
	# studyCorrelation déjà utilisé pour df principal
	# midtermScores: {"class_A": [...], "class_B": [...]} -> DataFrame long
	mid = _safe_get(raw_json, 'midtermScores')
	if isinstance(mid, dict) and 'class_A' in mid and 'class_B' in mid:
		class_A = list(mid['class_A'])
		class_B = list(mid['class_B'])
		midterm_df = pd.DataFrame({
			"classe": ['A'] * len(class_A) + ['B'] * len(class_B),
			"score": class_A + class_B
		})

	# approachGains: {"Approach1": [...], ...}
	app = _safe_get(raw_json, 'approachGains')
	if isinstance(app, dict):
		rows = []
		for k, v in app.items():
			if isinstance(v, list):
				for val in v:
					rows.append({"approche": k, "gain": val})
		if rows:
			approach_df = pd.DataFrame(rows)

	# courseFormat: {"presentiel": {"pass": x, "fail": y}, ...}
	cf = _safe_get(raw_json, 'courseFormat')
	if isinstance(cf, dict):
		rows = []
		for modality, results in cf.items():
			if isinstance(results, dict):
				for res_label, count in results.items():
					try:
						c = int(count)
					except Exception:
						continue
					rows.extend([{"modalite": modality, "resultat": res_label}] * c)
		if rows:
			course_format_df = pd.DataFrame(rows)

	# completionTimes: {"None": [...], "Simple": [...], "Advanced": [...]} -> DataFrame
	comp = _safe_get(raw_json, 'completionTimes')
	if isinstance(comp, dict):
		rows = []
		for level, vals in comp.items():
			if isinstance(vals, list):
				for tval in vals:
					rows.append({"niveau": level, "temps": tval})
		if rows:
			completion_df = pd.DataFrame(rows)

# Créer variables catégorielles dérivées si aucune catégorielle détectée sur df principal
if not cat_vars and {'hours', 'score'}.issubset(df.columns):
	try:
		df['hours_bin'] = pd.qcut(df['hours'], 2, labels=['Bas', 'Haut'])
		df['score_bin'] = pd.qcut(df['score'], 2, labels=['Faible', 'Élevé'])
		_, cat_vars = detect_variable_types(df)  # mettre à jour
	except Exception:
		pass

with st.expander("Aperçu des données", expanded=True):
	st.write("Dimensions :", df.shape)
	st.dataframe(df.head(20))
	st.write("Variables quantitatives détectées:", quant_vars or "(Aucune)")
	st.write("Variables qualitatives détectées:", cat_vars or "(Aucune)")


# ---------------------------------------------------------------------------
# Section Test t de Student
# ---------------------------------------------------------------------------
st.markdown("## 🔬 Tests principaux")
tests_tab = st.tabs(["t Student", "Corrélation", "Khi2", "Kruskal", "ANOVA"])

with tests_tab[0]:
	if not HAS_SCIPY:
		st.info("Tests indisponibles (SciPy manquant).")
	else:
		if len(cat_vars) >= 1 and len(quant_vars) >= 1:
			col1, col2 = st.columns(2)
			with col1:
				group = st.selectbox("Groupe (2 modalités)", cat_vars, key="t_group")
			with col2:
				val = st.selectbox("Variable quantitative", quant_vars, key="t_val")
			res = run_ttest(df, group, val)
			if res.get('ok'):
				c1, c2, c3 = st.columns(3)
				_metric_row((c1,c2,c3), [("t", f"{res['stat']:.3f}"),("p-value", f"{res['pvalue']:.3g}"),("Signif.", ("Oui" if res['pvalue']<alpha else "Non"))])
				st.markdown(_format_significance(res['pvalue'], alpha), unsafe_allow_html=True)
				st.pyplot(make_boxplot(df[df[group].isin(res['groups'])], group, val))
			else:
				st.warning(res.get('msg'))
		else:
			st.info("Ajouter / dériver une variable catégorielle pour ce test.")

with tests_tab[1]:
	if not HAS_SCIPY:
		st.info("Corrélation indisponible (SciPy manquant).")
	else:
		if len(quant_vars) >= 2:
			col1, col2 = st.columns(2)
			with col1:
				xv = st.selectbox("Quantitative 1", quant_vars, key="pear_x2")
			with col2:
				yv = st.selectbox("Quantitative 2", [q for q in quant_vars if q != xv], key="pear_y2")
			res = run_pearson(df, xv, yv)
			if res.get('ok'):
				c1, c2, c3 = st.columns(3)
				_metric_row((c1,c2,c3), [("r", f"{res['stat']:.3f}"),("p-value", f"{res['pvalue']:.3g}"),("n", res['n'])])
				st.markdown(_format_significance(res['pvalue'], alpha), unsafe_allow_html=True)
				st.pyplot(make_scatter(df, xv, yv))
			else:
				st.warning(res.get('msg'))
		else:
			st.info("Au moins deux quantitatives nécessaires.")

with tests_tab[2]:
	if not HAS_SCIPY:
		st.info("Test Khi2 indisponible (SciPy manquant).")
	else:
		if len(cat_vars) >= 2:
			col1, col2 = st.columns(2)
			with col1:
				c1v = st.selectbox("Qualitative 1", cat_vars, key="chi1")
			with col2:
				c2v = st.selectbox("Qualitative 2", [c for c in cat_vars if c != c1v], key="chi2")
			res = run_chi2(df, c1v, c2v)
			if res.get('ok'):
				c1c, c2c, c3c = st.columns(3)
				_metric_row((c1c,c2c,c3c), [("Chi2", f"{res['stat']:.3f}"),("p-value", f"{res['pvalue']:.3g}"),("ddl", res['dof'])])
				st.markdown(_format_significance(res['pvalue'], alpha), unsafe_allow_html=True)
				st.pyplot(make_heatmap(res['contingency']))
			else:
				st.warning(res.get('msg'))
		else:
			st.info("Deux qualitatives nécessaires.")

with tests_tab[3]:
	if not HAS_SCIPY:
		st.info("Kruskal indisponible (SciPy manquant).")
	else:
		if len(cat_vars) >= 1 and len(quant_vars) >= 1:
			group = st.selectbox("Groupe", cat_vars, key="kw_group2")
			val = st.selectbox("Quantitative", quant_vars, key="kw_val2")
			res = run_kruskal(df, group, val)
			if res.get('ok'):
				c1k,c2k,c3k = st.columns(3)
				_metric_row((c1k,c2k,c3k), [("H", f"{res['stat']:.3f}"),("p-value", f"{res['pvalue']:.3g}"),("k", res['k'])])
				st.markdown(_format_significance(res['pvalue'], alpha), unsafe_allow_html=True)
				st.pyplot(make_boxplot(df, group, val))
			else:
				st.warning(res.get('msg'))
		else:
			st.info("Variables insuffisantes.")

with tests_tab[4]:
	if not HAS_SCIPY:
		st.info("ANOVA indisponible (SciPy manquant).")
	else:
		if len(cat_vars) >= 1 and len(quant_vars) >= 1:
			group = st.selectbox("Groupe", cat_vars, key="an_group2")
			val = st.selectbox("Quantitative", quant_vars, key="an_val2")
			res = run_anova(df, group, val)
			if res.get('ok'):
				c1a,c2a,c3a = st.columns(3)
				_metric_row((c1a,c2a,c3a), [("F", f"{res['stat']:.3f}"),("p-value", f"{res['pvalue']:.3g}"),("k", res['k'])])
				st.markdown(_format_significance(res['pvalue'], alpha), unsafe_allow_html=True)
				st.pyplot(make_boxplot(df, group, val))
			else:
				st.warning(res.get('msg'))
		else:
			st.info("Variables insuffisantes.")


# ---------------------------------------------------------------------------
# Exploration additionnelle optionnelle
# ---------------------------------------------------------------------------
st.markdown("## 🔎 Exploration libre")
if quant_vars:
	c1, c2 = st.columns([2,1])
	with c2:
		dist_col = st.selectbox("Variable", quant_vars, key="dist_col")
	with c1:
		st.pyplot(make_distplot(df, dist_col))
else:
	st.info("Aucune quantitative détectée.")

# ---------------------------------------------------------------------------
# Sections additionnelles dérivées du notebook (données structurées)
# ---------------------------------------------------------------------------
st.markdown("## 🧩 Sous-jeux spécifiques")

colA, colB = st.columns(2)
with colA:
	st.subheader("Midterm Scores (Test t / ANOVA)")
	if midterm_df is not None:
		st.caption(f"Taille: {midterm_df.shape[0]} observations | Colonnes: {list(midterm_df.columns)}")
		res_mid_t = run_ttest(midterm_df.rename(columns={'classe':'classe'}), 'classe', 'score')
		if res_mid_t.get('ok'):
			st.write(f"t = {res_mid_t['stat']:.3f} | p = {res_mid_t['pvalue']:.3g}")
			st.info(interpret_pvalue(res_mid_t['pvalue'], alpha))
			st.pyplot(make_boxplot(midterm_df, 'classe', 'score'))
		else:
			st.warning(res_mid_t.get('msg','Test t impossible.'))
	else:
		st.info("Données midtermScores indisponibles dans le JSON.")

with colB:
	st.subheader("Approach Gains (ANOVA & Kruskal)")
	if approach_df is not None:
		st.caption(f"Groupes: {approach_df['approche'].nunique()} | n={approach_df.shape[0]}")
		res_app_an = run_anova(approach_df, 'approche', 'gain')
		if res_app_an.get('ok'):
			st.write(f"ANOVA: F = {res_app_an['stat']:.3f} | p = {res_app_an['pvalue']:.3g}")
			st.info(interpret_pvalue(res_app_an['pvalue'], alpha))
		res_app_kw = run_kruskal(approach_df, 'approche', 'gain')
		if res_app_kw.get('ok'):
			st.write(f"Kruskal: H = {res_app_kw['stat']:.3f} | p = {res_app_kw['pvalue']:.3g}")
			st.info(interpret_pvalue(res_app_kw['pvalue'], alpha))
		st.pyplot(make_boxplot(approach_df, 'approche', 'gain'))
	else:
		st.info("Données approachGains indisponibles.")

colC, colD = st.columns(2)
with colC:
	st.subheader("Course Format (Khi2)")
	if course_format_df is not None and not course_format_df.empty:
		cont = pd.crosstab(course_format_df['modalite'], course_format_df['resultat'])
		chi2, p, dof, exp = chi2_contingency(cont)
		st.write(f"Chi2 = {chi2:.3f} | ddl = {dof} | p = {p:.3g}")
		st.info(interpret_pvalue(p, alpha))
		st.pyplot(make_heatmap(cont))
		st.pyplot(make_course_format_bar(cont))
	else:
		st.info("Données courseFormat indisponibles.")

with colD:
	st.subheader("Completion Times (ANOVA & Kruskal)")
	if completion_df is not None and not completion_df.empty:
		res_comp_an = run_anova(completion_df, 'niveau', 'temps')
		if res_comp_an.get('ok'):
			st.write(f"ANOVA: F = {res_comp_an['stat']:.3f} | p = {res_comp_an['pvalue']:.3g}")
			st.info(interpret_pvalue(res_comp_an['pvalue'], alpha))
		res_comp_kw = run_kruskal(completion_df, 'niveau', 'temps')
		if res_comp_kw.get('ok'):
			st.write(f"Kruskal: H = {res_comp_kw['stat']:.3f} | p = {res_comp_kw['pvalue']:.3g}")
			st.info(interpret_pvalue(res_comp_kw['pvalue'], alpha))
		st.pyplot(make_boxplot(completion_df, 'niveau', 'temps'))
		with st.expander("Distributions par niveau"):
			for lvl, sub in completion_df.groupby('niveau'):
				st.pyplot(make_distplot(sub, 'temps'))
	else:
		st.info("Données completionTimes indisponibles.")

# ---------------------------------------------------------------------------
# 8. Visualisation spécifique StudyCorrelation (régression)
# ---------------------------------------------------------------------------
if {'hours','score'}.issubset(df.columns):
	st.markdown("## 📈 Heures d'étude vs Score")
	reg_df = df[['hours','score']].dropna()
	if not reg_df.empty:
		st.write(f"Observations valides: {len(reg_df)}")
		fig_reg = make_regression(reg_df, 'hours', 'score')
		st.pyplot(fig_reg)
		# Corrélation
		try:
			corr, p_corr = pearsonr(reg_df['hours'], reg_df['score'])
			st.write(f"Corrélation de Pearson r = {corr:.3f} | p = {p_corr:.3g}")
			st.info(interpret_pvalue(p_corr, alpha))
		except Exception:
			st.warning("Impossible de calculer la corrélation.")
		with st.expander("Histogrammes séparés"):
			c1, c2 = st.columns(2)
			with c1:
				st.pyplot(make_distplot(reg_df, 'hours'))
			with c2:
				st.pyplot(make_distplot(reg_df, 'score'))
else:
	st.info("Colonnes 'hours' et 'score' non disponibles pour la régression.")

# ---------------------------------------------------------------------------
# 9. Visualisations supplémentaires Midterm (hist + violin)
# ---------------------------------------------------------------------------
if midterm_df is not None:
	st.markdown("## 🎯 Midterm Scores - Visualisations supplémentaires")
	A_vals = midterm_df[midterm_df['classe']=='A']['score'].tolist()
	B_vals = midterm_df[midterm_df['classe']=='B']['score'].tolist()
	if A_vals and B_vals:
		tab1, tab2, tab3 = st.tabs(["Histogrammes", "Violin", "Récap Stats"])
		with tab1:
			st.pyplot(make_side_by_side_hists(A_vals, B_vals, labels=("Classe A","Classe B")))
		with tab2:
			st.pyplot(make_violin_two(A_vals, B_vals, labels=("Classe A","Classe B")))
		with tab3:
			st.write(pd.DataFrame({
				"Classe": ["A","B"],
				"n": [len(A_vals), len(B_vals)],
				"Moyenne": [np.mean(A_vals), np.mean(B_vals)],
				"Écart-type": [np.std(A_vals, ddof=1), np.std(B_vals, ddof=1)]
			}))


# ---------------------------------------------------------------------------
# Notes pédagogiques
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Notes pédagogiques"):
	st.markdown(
		textwrap.dedent(
			f"""
			**Rappels :**
			- Le seuil α (ici configurable) représente le risque d'erreur de Type I.
			- Une p-value < α => résultat statistiquement significatif.
			- Test t suppose normalité et variances homogènes (ici *Welch* pour robustesse).
			- ANOVA suppose normalité intra-groupes & homogénéité des variances.
			- Kruskal-Wallis est l'alternative non paramétrique (basée sur rangs).
			- Khi2 : effectifs attendus idéalement >= 5 (sinon prudence dans l'interprétation).
			- Corrélation de Pearson mesure une relation linéaire (attention aux outliers et causalité!).
			"""
		)
	)

st.success("Analyse terminée. Ajustez les paramètres pour explorer davantage.")

# Fin du script
