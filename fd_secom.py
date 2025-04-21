import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"D:\\ML Projects\Fault Detection SECOM\uci-secom.csv",low_memory=False)
df = df.iloc[1:].reset_index(drop=True).drop(columns=["Column1"])
df["Column592"] = df["Column592"].astype(int).replace(-1, 0)

X = df.drop(columns=["Column592"]).apply(pd.to_numeric, errors='coerce')
y = df["Column592"]

X = SimpleImputer(strategy="mean").fit_transform(X)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

test_model(model, X_test, y_test)
