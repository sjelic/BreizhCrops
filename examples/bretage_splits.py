import numpy as np
from sklearn.model_selection import train_test_split

X_fr = np.load('../data/X.npy')
y_fr = np.load('../data/y.npy')

X_0102, X_0304, y_0102, y_0304 = train_test_split(X_fr, y_fr, test_size=0.5, random_state=1, shuffle=True, stratify=y_fr)
X_01, X_02, y_01, y_02 = train_test_split(X_0102, y_0102, test_size=0.5, random_state=1, shuffle=True, stratify=y_0102)
X_03, X_04, y_03, y_04 = train_test_split(X_0304, y_0304, test_size=0.5, random_state=1, shuffle=True, stratify=y_0304)
np.save('../data/X_fr_01.npy', X_01)
np.save('../data/X_fr_02.npy', X_02)
np.save('../data/X_fr_03.npy', X_03)
np.save('../data/X_fr_04.npy', X_04)

np.save('../data/y_fr_01.npy', y_01)
np.save('../data/y_fr_02.npy', y_02)
np.save('../data/y_fr_03.npy', y_03)
np.save('../data/y_fr_04.npy', y_04)