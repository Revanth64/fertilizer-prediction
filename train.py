import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from category_encoders import BinaryEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Load dataset
fertiliser = pd.read_csv('fertilizer_recommendation.csv')
df = fertiliser.copy()
df.columns = df.columns.str.lower()

cat_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
num_columns = [col for col in df.columns if col not in cat_columns and col != 'recommended_fertilizer']

strat_split = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.2)
for train_index, test_index in strat_split.split(df, df['recommended_fertilizer']):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]

bin_columns = ['crop_type', 'previous_crop']
ord_columns = ['crop_growth_stage', 'irrigation_type']
ord_precedence = [['Sowing', 'Flowering', 'Vegetative', 'Harvest'], ['Rainfed', 'Drip', 'Sprinkler', 'Canal']]
onehot_columns = ['season', 'region', 'soil_type']

column_transformer = ColumnTransformer([
    ('binary', BinaryEncoder(), bin_columns),
    ('ordinal', OrdinalEncoder(categories=ord_precedence), ord_columns),
    ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_columns),
    ('scale', StandardScaler(), num_columns)
])

# Create robust Pipeline directly
pipe_forest = Pipeline([
    ('encode', column_transformer),
    ('forest', RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=42))
])

# Fit on the entire training set
print("Training the pipeline...")
X_train = strat_train_set.drop('recommended_fertilizer', axis=1)
y_train = strat_train_set['recommended_fertilizer']
pipe_forest.fit(X_train, y_train)

# Save the Pipeline (which embeds the transformers) as the main model file
joblib.dump(pipe_forest, 'random_forest_model.joblib')
print("Random forest pipeline saved successfully!")
