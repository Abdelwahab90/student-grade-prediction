"""
Student Performance Prediction 

"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve,
                              confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
import os
import tkinter as tk
from tkinter import filedialog


def select_data_file():
    
    filetypes = [
        ('Excel files', '*.xlsx *.xls'),
        ('CSV files', '*.csv'),
        ('All files', '*.*'),
    ]
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title='Select student performance dataset',
            filetypes=filetypes,
        )
        root.destroy()
    except tk.TclError:
        path = ''

    if path:
        return path

    print('No file selected in the dialog.')
    path = input('Enter the full path to your data file: ').strip().strip('"').strip("'")
    if not path or not os.path.isfile(path):
        raise SystemExit('No valid data file provided. Exiting.')
    return path


def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if ext == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported file format: {ext}. Use .xlsx, .xls, or .csv')


DATA_FILE = select_data_file()
DATA_DIR = os.path.dirname(os.path.abspath(DATA_FILE))
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')
RESULTS_CSV = os.path.join(DATA_DIR, 'model_results.csv')

os.makedirs(FIGURES_DIR, exist_ok=True)
print(f'Using dataset: {DATA_FILE}')
print(f'Figures will be saved to: {FIGURES_DIR}')

# =============================================================================
# 2. LOAD DATA
# =============================================================================
df = load_dataset(DATA_FILE)
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nGrade Distribution:")
print(df['grade'].value_counts())

# =============================================================================
# 3. DATA QUALITY REPORT VISUALIZATIONS
# =============================================================================

# Figure 1: Grade Distribution (Class Imbalance)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
grade_counts = df['grade'].value_counts()
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
axes[0].bar(grade_counts.index, grade_counts.values, color=colors[:len(grade_counts)])
axes[0].set_title('Grade Distribution (Before Balancing)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Grade')
axes[0].set_ylabel('Count')
for i, v in enumerate(grade_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%',
            colors=colors[:len(grade_counts)], startangle=90)
axes[1].set_title('Grade Distribution (%)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/01_grade_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 2: Numerical Feature Distributions
num_cols = ['study_hours_per_day', 'attendance_percentage', 'assignment_score',
            'midterm_score', 'final_exam_score', 'participation_score',
            'sleep_hours']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='#3498db', edgecolor='white', alpha=0.8)
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=10)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
for j in range(len(num_cols), len(axes)):
    axes[j].axis('off')
plt.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Boxplots by Grade
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
grade_order = ['A', 'B', 'C', 'D', 'F']
for i, col in enumerate(num_cols):
    df.boxplot(column=col, by='grade', ax=axes[i], grid=False)
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=9)
    axes[i].set_xlabel('Grade')
    axes[i].set_ylabel('')
for j in range(len(num_cols), len(axes)):
    axes[j].axis('off')
plt.suptitle('Feature Distribution by Grade', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/03_boxplots_by_grade.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, annot_kws={'size': 9})
ax.set_title('Correlation Heatmap of Numerical Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: Categorical features vs Grade
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cat_cols = ['gender', 'internet_access', 'extra_classes']
for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['grade'])
    ct[['A', 'B', 'C', 'D', 'F']].plot(kind='bar', ax=axes[i], colormap='Set2', edgecolor='white')
    axes[i].set_title(f'{col.replace("_", " ").title()} vs Grade', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(col.replace('_', ' ').title())
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].legend(title='Grade', fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/05_categorical_vs_grade.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[INFO] Data quality figures saved.")

# =============================================================================
# 4. DATA PREPROCESSING
# =============================================================================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

df_clean = df.copy()

# Drop student_id (not predictive)
df_clean.drop(columns=['student_id'], inplace=True)

# Check for duplicates
dups = df_clean.duplicated().sum()
print(f"Duplicate rows: {dups}")

# Encode categorical features
le_gender = LabelEncoder()
le_internet = LabelEncoder()
le_extra = LabelEncoder()
le_parent = LabelEncoder()

df_clean['gender_enc'] = le_gender.fit_transform(df_clean['gender'])          # Female=0, Male=1
df_clean['internet_enc'] = le_internet.fit_transform(df_clean['internet_access'])  # No=0, Yes=1
df_clean['extra_enc'] = le_extra.fit_transform(df_clean['extra_classes'])         # No=0, Yes=1
df_clean['parent_enc'] = le_parent.fit_transform(df_clean['parent_education'])    # Ordinal encoding

print(f"\nGender encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"Internet encoding: {dict(zip(le_internet.classes_, le_internet.transform(le_internet.classes_)))}")
print(f"Extra classes encoding: {dict(zip(le_extra.classes_, le_extra.transform(le_extra.classes_)))}")
print(f"Parent education encoding: {dict(zip(le_parent.classes_, le_parent.transform(le_parent.classes_)))}")

# Drop original string columns
df_clean.drop(columns=['gender', 'internet_access', 'extra_classes', 'parent_education'], inplace=True)

# Encode target
le_grade = LabelEncoder()
df_clean['grade_enc'] = le_grade.fit_transform(df_clean['grade'])
print(f"\nGrade encoding: {dict(zip(le_grade.classes_, le_grade.transform(le_grade.classes_)))}")
# A=0, B=1, C=2, D=3, F=4

# Feature matrix and target
feature_cols = [c for c in df_clean.columns if c not in ['grade', 'grade_enc']]
X = df_clean[feature_cols]
y = df_clean['grade_enc']

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
print(f"\nClass distribution before SMOTE:")
for g, c in zip(le_grade.classes_, np.bincount(y)):
    print(f"  Grade {g}: {c}")

# Outlier detection and report (IQR method)
print("\nOutlier Analysis (IQR method):")
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col}: {outliers} outliers")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# =============================================================================
# 5. CLASS BALANCING WITH SMOTE
# =============================================================================
print("\n" + "="*60)
print("CLASS BALANCING (SMOTE)")
print("="*60)

# The dataset is highly imbalanced: A=12, F=8, vs C=503
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"Before SMOTE: {X_scaled.shape[0]} samples")
print(f"After SMOTE:  {X_resampled.shape[0]} samples")
print("Class distribution after SMOTE:")
for i, cls in enumerate(le_grade.classes_):
    count = (y_resampled == i).sum()
    print(f"  Grade {cls} (encoded {i}): {count}")

# Figure 6: Before vs After SMOTE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
before_counts = pd.Series(y).value_counts().sort_index()
after_counts = pd.Series(y_resampled).value_counts().sort_index()
grade_labels = le_grade.classes_

axes[0].bar(grade_labels, [before_counts.get(i, 0) for i in range(len(grade_labels))],
            color='#e74c3c', edgecolor='white')
axes[0].set_title('Class Distribution Before SMOTE', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Grade')
axes[0].set_ylabel('Count')
for i, v in enumerate([before_counts.get(i, 0) for i in range(len(grade_labels))]):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

axes[1].bar(grade_labels, [after_counts.get(i, 0) for i in range(len(grade_labels))],
            color='#2ecc71', edgecolor='white')
axes[1].set_title('Class Distribution After SMOTE', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Grade')
axes[1].set_ylabel('Count')
for i, v in enumerate([after_counts.get(i, 0) for i in range(len(grade_labels))]):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR + '/06_smote_balancing.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 6. MACHINE LEARNING MODELING
# =============================================================================
print("\n" + "="*60)
print("MACHINE LEARNING MODELING")
print("="*60)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

models = {
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
}

results = {}
y_proba_dict = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation scores
    acc_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv,
                                  scoring='accuracy', n_jobs=1)
    prec_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv,
                                   scoring='precision_weighted', n_jobs=1)
    rec_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv,
                                  scoring='recall_weighted', n_jobs=1)
    f1_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv,
                                 scoring='f1_weighted', n_jobs=1)
    
    results[name] = {
        'Accuracy': acc_scores.mean(),
        'Accuracy_std': acc_scores.std(),
        'Precision': prec_scores.mean(),
        'Recall': rec_scores.mean(),
        'F1-Score': f1_scores.mean(),
    }
    
    # For ROC curve: fit on all resampled data and predict on original data
    model.fit(X_resampled, y_resampled)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_scaled)
        y_proba_dict[name] = y_proba
    
    print(f"  Accuracy:  {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"  Precision: {prec_scores.mean():.4f}")
    print(f"  Recall:    {rec_scores.mean():.4f}")
    print(f"  F1-Score:  {f1_scores.mean():.4f}")

# =============================================================================
# 7. RESULTS TABLE
# =============================================================================
print("\n" + "="*60)
print("MODEL COMPARISON TABLE")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Accuracy_std']]
results_df = results_df.round(4)
print(results_df.to_string())

# Best model
best_model_name = results_df['Accuracy'].idxmax()
print(f"\nBest Model: {best_model_name} (Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f})")

# =============================================================================
# 8. VISUALIZATION: MODEL COMPARISON BAR CHART
# =============================================================================

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(models))
width = 0.18
colors_models = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

fig, ax = plt.subplots(figsize=(14, 6))
for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in models.keys()]
    bars = ax.bar(x + i * width, values, width, label=metric, color=colors_models[i], alpha=0.85, edgecolor='white')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison (10-Fold Cross-Validation)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(list(models.keys()), fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/07_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 9. ROC CURVE (One-vs-Rest, macro average per model)
# =============================================================================
from sklearn.preprocessing import label_binarize
from itertools import cycle

n_classes = len(le_grade.classes_)
y_bin = label_binarize(y, classes=list(range(n_classes)))

roc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
roc_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
roc_linewidths = [2.5, 2.5, 2.5, 2.5, 2.5]

plt.figure(figsize=(10, 8))
ax = plt.gca()

for idx, (name, y_proba) in enumerate(y_proba_dict.items()):
    fpr_list, tpr_list = [], []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
    mean_tpr /= n_classes
    
    auc_val = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
    results[name]['AUC'] = auc_val
    
    plt.plot(all_fpr, mean_tpr,
             label=f'{name} (AUC = {auc_val:.3f})',
             color=roc_colors[idx % len(roc_colors)],
             linestyle=roc_linestyles[idx % len(roc_linestyles)],
             linewidth=roc_linewidths[idx % len(roc_linewidths)])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC=0.5)')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (Macro-Average, One-vs-Rest)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/08_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
# =============================================================================
# 10. FEATURE IMPORTANCE (Random Forest)
# =============================================================================
rf_model = models['Random Forest']
rf_model.fit(X_resampled, y_resampled)
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind='bar', ax=ax, color='#3498db', edgecolor='white', alpha=0.85)
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Feature')
ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=45)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 11. CONFUSION MATRICES — ALL 5 MODELS
# =============================================================================
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_grade.classes_,
                yticklabels=le_grade.classes_, ax=ax,
                annot_kws={'size': 9})
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

plt.suptitle('Confusion Matrices — All Five Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR + '/10_confusion_matrices_all.png', dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 12. FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
final_df = pd.DataFrame(results).T
print(final_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']].round(4).to_string())
print(f"\nAll figures saved to {FIGURES_DIR}")
print("Analysis complete!")

# Save results to CSV for report
final_df.to_csv(RESULTS_CSV)
print(f"Results saved to {RESULTS_CSV}")

# =============================================================================
# 13. PREDICTION ON NEW STUDENTS
# =============================================================================
print("\n" + "="*60)
print("PREDICTION ON NEW STUDENTS")
print("="*60)

# ------------------------------------------------------------------
# Use the best model (Random Forest) for prediction
# ------------------------------------------------------------------
best_model = models['Random Forest']   # already fitted on X_resampled

# ------------------------------------------------------------------
# Helper: encode a raw input dict → scaled feature vector
# ------------------------------------------------------------------
def encode_student(raw: dict) -> np.ndarray:
    """
    raw keys (same as original dataset columns, minus student_id & grade):
        gender              : 'Male' | 'Female'
        study_hours_per_day : float  (1.0 – 10.0)
        attendance_percentage: float (0 – 100)
        assignment_score    : float  (0 – 100)
        midterm_score       : float  (0 – 100)
        final_exam_score    : float  (0 – 100)
        participation_score : float  (0 – 100)
        internet_access     : 'Yes' | 'No'
        extra_classes       : 'Yes' | 'No'
        parent_education    : 'High School' | 'Bachelor' | 'Master' | 'PhD'
        sleep_hours         : float  (4.0 – 9.0)
    """
    gender_map      = {'Female': 0, 'Male': 1}
    internet_map    = {'No': 0, 'Yes': 1}
    extra_map       = {'No': 0, 'Yes': 1}
    parent_map      = {'Bachelor': 0, 'High School': 1, 'Master': 2, 'PhD': 3}

    row = [
        raw['study_hours_per_day'],
        raw['attendance_percentage'],
        raw['assignment_score'],
        raw['midterm_score'],
        raw['final_exam_score'],
        raw['participation_score'],
        raw['sleep_hours'],
        gender_map[raw['gender']],
        internet_map[raw['internet_access']],
        extra_map[raw['extra_classes']],
        parent_map[raw['parent_education']],
    ]
    # scale using the same scaler fitted on training data
    return scaler.transform([row])

# ------------------------------------------------------------------
# Example students to predict
# ------------------------------------------------------------------
new_students = [
    {
        'name': 'Ahmed Mohamed',
        'gender': 'Male',
        'study_hours_per_day': 7.5,
        'attendance_percentage': 90.0,
        'assignment_score': 85.0,
        'midterm_score': 78.0,
        'final_exam_score': 88.0,
        'participation_score': 70.0,
        'internet_access': 'Yes',
        'extra_classes': 'Yes',
        'parent_education': 'Master',
        'sleep_hours': 7.0,
    },
    {
        'name': 'Sara Ali',
        'gender': 'Female',
        'study_hours_per_day': 2.0,
        'attendance_percentage': 45.0,
        'assignment_score': 35.0,
        'midterm_score': 30.0,
        'final_exam_score': 32.0,
        'participation_score': 15.0,
        'internet_access': 'No',
        'extra_classes': 'No',
        'parent_education': 'High School',
        'sleep_hours': 5.0,
    },
    {
        'name': 'Omar Hassan',
        'gender': 'Male',
        'study_hours_per_day': 5.0,
        'attendance_percentage': 70.0,
        'assignment_score': 65.0,
        'midterm_score': 60.0,
        'final_exam_score': 67.0,
        'participation_score': 55.0,
        'internet_access': 'Yes',
        'extra_classes': 'No',
        'parent_education': 'Bachelor',
        'sleep_hours': 6.5,
    },
]

print(f"\n{'Student':<18} {'Predicted Grade':<18} {'Confidence':>12}   Probabilities per Grade")
print("-" * 80)

for student in new_students:
    name = student.pop('name')
    X_new = encode_student(student)

    pred_encoded  = best_model.predict(X_new)[0]
    pred_grade    = le_grade.inverse_transform([pred_encoded])[0]
    proba         = best_model.predict_proba(X_new)[0]
    confidence    = proba.max() * 100

    prob_str = '  '.join(
        [f"{le_grade.classes_[i]}:{proba[i]*100:5.1f}%"
         for i in range(len(le_grade.classes_))]
    )
    print(f"{name:<18} Grade {pred_grade:<12} {confidence:>10.1f}%   {prob_str}")
    student['name'] = name   # restore name

print("\n[INFO] Prediction complete.")
print("[TIP]  To predict your own student, edit the 'new_students' list above.")
import joblib

joblib.dump(best_model, os.path.join(DATA_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
joblib.dump(le_grade, os.path.join(DATA_DIR, "label_encoder.pkl"))