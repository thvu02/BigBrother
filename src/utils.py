import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import config

def classify_income(income_value):
    if income_value < config.INCOME_THRESHOLDS['low']:
        return config.INCOME_LABELS['low']
    elif income_value < config.INCOME_THRESHOLDS['medium']:
        return config.INCOME_LABELS['medium']
    elif income_value < config.INCOME_THRESHOLDS['high']:
        return config.INCOME_LABELS['high']
    else:
        return config.INCOME_LABELS['very_high']

# link ad user IDs to census person IDs using quasi identifiers
def reidentification_attack(ads, census):
    results = []
    user_targeting = ads.groupby('user_id').agg({
        'target_ages': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'target_gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        'target_locations': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    use_protected = 'age_protected' in census.columns and 'zip_protected' in census.columns

    if use_protected:
        # Detect generalization level by examining a sample ZIP code
        sample_zip = census['zip_protected'].iloc[0]
        # Count the number of 'X's to determine generalization level
        num_x = sample_zip.count('X')

        # Detect if using coarse age groups (3 groups) vs standard (6 groups)
        unique_age_groups = census['age_protected'].unique()
        use_coarse_age = len(unique_age_groups) <= 3

        # Use protected quasi-identifiers (generalized)
        for _, user in user_targeting.iterrows():
            age_group = user['target_ages']
            gender = user['target_gender']

            # Map standard age group to coarse age group if needed
            if use_coarse_age:
                # Map standard age groups to coarse age groups
                age_mapping = {
                    '18-24': '18-34',
                    '25-34': '18-34',
                    '35-44': '35-54',
                    '45-54': '35-54',
                    '55-64': '55+',
                    '65+': '55+'
                }
                age_group = age_mapping.get(age_group, age_group)

            # Generate ZIP pattern based on generalization level
            if num_x == 3:  # 2-digit ZIP (e.g., "90XXX")
                zip_pattern = str(user['target_locations'])[:2] + 'XXX'
            else:  # 2 X's means 3-digit ZIP (e.g., "900XX")
                zip_pattern = str(user['target_locations'])[:3] + 'XX'

            matches = census[
                (census['age_protected'] == age_group) &
                (census['gender'] == gender) &
                (census['zip_protected'] == zip_pattern)
            ]

            predicted = matches.iloc[0]['person_id'] if len(matches) > 0 else None
            results.append({
                'user_id': user['user_id'],
                'k_anonymity': len(matches),
                'correct': predicted == user['user_id']
            })
    else:
        # Use original quasi-identifiers (exact matching)
        user_targeting['zip_prefix'] = user_targeting['target_locations'].astype(str).str[:4]

        for _, user in user_targeting.iterrows():
            age_min, age_max = config.AGE_GROUPS_STANDARD[user['target_ages']]

            census_zip = census['zip_code_enhanced'].astype(str).str[:4]
            matches = census[
                (census['age'] >= age_min) & (census['age'] <= age_max) &
                (census['gender'] == user['target_gender']) & (census_zip == user['zip_prefix'])
            ]

            predicted = matches.iloc[0]['person_id'] if len(matches) > 0 else None
            results.append({
                'user_id': user['user_id'],
                'k_anonymity': len(matches),
                'correct': predicted == user['user_id']
            })

    return pd.DataFrame(results)

# test reconstruction attack on different ML models (Random Forest, Gradient Boosting, SVM, NN)
def test_all_models(X, census, attr, transform=None):
    census_indexed = census.set_index('person_id')
    y = census_indexed.loc[X.index][attr]
    if transform:
        y = y.apply(transform)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE, stratify=y
    )

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.ML_RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=config.GB_N_ESTIMATORS,
            max_depth=config.GB_MAX_DEPTH,
            random_state=config.ML_RANDOM_STATE
        ),
        'SVM': SVC(kernel='rbf', random_state=config.ML_RANDOM_STATE),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=config.NN_HIDDEN_LAYERS,
            max_iter=config.NN_MAX_ITER,
            random_state=config.ML_RANDOM_STATE
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    return results

# test reconstruction attacks on different ML models using DP-protected data
def test_all_models_with_dp(X, census_protected, baseline_results):
    results = {'income': {}, 'education': {}, 'occupation': {}}

    for attr in ['income', 'education', 'occupation']:
        attr_protected = attr + '_protected'

        try:
            census_indexed = census_protected.set_index('person_id')
            X_filtered = X.loc[X.index.isin(census_indexed.index)]
            y = census_indexed.loc[X_filtered.index][attr_protected]

            if len(y) < 100:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y, test_size=config.ML_TEST_SIZE,
                random_state=config.ML_RANDOM_STATE, stratify=y
            )

            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH,
                    random_state=config.ML_RANDOM_STATE
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=config.GB_N_ESTIMATORS,
                    max_depth=config.GB_MAX_DEPTH,
                    random_state=config.ML_RANDOM_STATE
                ),
                'SVM': SVC(kernel='rbf', random_state=config.ML_RANDOM_STATE),
                'Neural Network': MLPClassifier(
                    hidden_layer_sizes=config.NN_HIDDEN_LAYERS,
                    max_iter=config.NN_MAX_ITER,
                    random_state=config.ML_RANDOM_STATE
                )
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[attr][name] = accuracy_score(y_test, y_pred)
        except:
            pass

    return results

# laplace + randomized response + generalization DP
# generate updated census dataset with protected columns
def apply_original_dp(census, epsilon=None):
    if epsilon is None:
        epsilon = config.EPSILON

    protected = census.copy()

    # laplace noise for income
    income_scale = config.INCOME_SENSITIVITY / epsilon
    income_noise = np.random.laplace(0, income_scale, size=len(protected))
    protected['income_noisy'] = (protected['income'] + income_noise).clip(0, 500000).round().astype(int)
    protected['income_protected'] = protected['income_noisy'].apply(classify_income)

    # randomized response for education and occupation
    p_true = np.exp(epsilon) / (np.exp(epsilon) + 1)

    edu_cats = census['education'].unique()
    edu_mask = np.random.random(len(protected)) >= p_true
    protected['education_protected'] = protected['education'].copy()
    protected.loc[edu_mask, 'education_protected'] = np.random.choice(edu_cats, edu_mask.sum())

    occ_cats = protected['occupation'].unique()
    occ_mask = np.random.random(len(protected)) >= p_true
    protected['occupation_protected'] = protected['occupation'].copy()
    protected.loc[occ_mask, 'occupation_protected'] = np.random.choice(occ_cats, occ_mask.sum())

    # generalization
    zip_digits = config.ZIP_GENERALIZATION['original']
    protected['zip_protected'] = protected['zip_code_enhanced'].astype(str).str[:zip_digits] + 'X' * (5 - zip_digits)
    protected['age_protected'] = protected['age_group']

    return protected

# adaptive DP; budget allocation based on sensitivity
# create protected census dataset with optimized budget allocation
def apply_adaptive_dp(census, total_epsilon=None):
    if total_epsilon is None:
        total_epsilon = config.EPSILON

    protected = census.copy()

    # income with adaptive budget
    eps_income = total_epsilon * config.ADAPTIVE_BUDGET['income']
    income_scale = config.INCOME_SENSITIVITY / eps_income
    income_noise = np.random.laplace(0, income_scale, size=len(protected))
    protected['income_protected'] = (protected['income'] + income_noise).clip(0, 500000).round().astype(int)
    protected['income_protected'] = protected['income_protected'].apply(classify_income)

    # education with adaptive budget
    eps_edu = total_epsilon * config.ADAPTIVE_BUDGET['education']
    p_true_edu = np.exp(eps_edu) / (np.exp(eps_edu) + 1)
    edu_cats = census['education'].unique()
    edu_mask = np.random.random(len(protected)) >= p_true_edu
    protected['education_protected'] = protected['education'].copy()
    protected.loc[edu_mask, 'education_protected'] = np.random.choice(edu_cats, edu_mask.sum())

    # occupation with adaptive budget
    eps_occ = total_epsilon * config.ADAPTIVE_BUDGET['occupation']
    p_true_occ = np.exp(eps_occ) / (np.exp(eps_occ) + 1)
    occ_cats = protected['occupation'].unique()
    occ_mask = np.random.random(len(protected)) >= p_true_occ
    protected['occupation_protected'] = protected['occupation'].copy()
    protected.loc[occ_mask, 'occupation_protected'] = np.random.choice(occ_cats, occ_mask.sum())

    # generalization
    zip_digits = config.ZIP_GENERALIZATION['original']
    protected['zip_protected'] = protected['zip_code_enhanced'].astype(str).str[:zip_digits] + 'X' * (5 - zip_digits)
    protected['age_protected'] = protected['age_group']

    return protected

# multi-layer DP; gaussian + generalization + k-suppression
# create protected census dataset with multiple defense layers
def apply_multilayer_dp(census, epsilon=None, delta=None, k_threshold=None):
    if epsilon is None:
        epsilon = config.EPSILON
    if delta is None:
        delta = config.DELTA
    if k_threshold is None:
        k_threshold = config.K_ANONYMITY_THRESHOLD

    protected = census.copy()

    # split budget
    eps_income = 0.25 * epsilon
    eps_categorical = 0.25 * epsilon

    # Gaussian noise for income
    noise_scale = config.INCOME_SENSITIVITY_GAUSSIAN * np.sqrt(2 * np.log(1.25 / delta)) / eps_income
    income_noise = np.random.normal(0, noise_scale, size=len(protected))
    protected['income_protected'] = (protected['income'] + income_noise).clip(0, 500000).round().astype(int)
    protected['income_protected'] = protected['income_protected'].apply(classify_income)

    # Randomized response
    p_true = np.exp(eps_categorical) / (np.exp(eps_categorical) + 1)

    edu_cats = census['education'].unique()
    edu_mask = np.random.random(len(protected)) >= p_true
    protected['education_protected'] = protected['education'].copy()
    protected.loc[edu_mask, 'education_protected'] = np.random.choice(edu_cats, edu_mask.sum())

    occ_cats = protected['occupation'].unique()
    occ_mask = np.random.random(len(protected)) >= p_true
    protected['occupation_protected'] = protected['occupation'].copy()
    protected.loc[occ_mask, 'occupation_protected'] = np.random.choice(occ_cats, occ_mask.sum())

    # Enhanced generalization
    zip_digits = config.ZIP_GENERALIZATION['multilayer']
    protected['zip_protected'] = protected['zip_code_enhanced'].astype(str).str[:zip_digits] + 'X' * (5 - zip_digits)

    def classify_age_coarse(age):
        for group, (min_age, max_age) in config.AGE_GROUPS_COARSE.items():
            if min_age <= age <= max_age:
                return group
        return '55+'

    protected['age_protected'] = protected['age'].apply(classify_age_coarse)

    # k-anonymity suppression
    quasi_id_cols = ['age_protected', 'gender', 'zip_protected']
    k_counts = protected.groupby(quasi_id_cols).size()
    protected['k_value'] = protected[quasi_id_cols].apply(
        lambda row: k_counts.get(tuple(row), 0), axis=1
    )
    suppressed = protected[protected['k_value'] >= k_threshold].copy()

    return suppressed

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DP-SGD implementation using Opacus
class DPNeuralNetwork:
    def __init__(self, epsilon=None, delta=None, clip_norm=None):
        self.epsilon = epsilon if epsilon is not None else config.EPSILON
        self.delta = delta if delta is not None else config.DELTA
        self.clip_norm = clip_norm if clip_norm is not None else config.DPSGD_CLIP_NORM
        self.model = None
        self.label_encoder = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X_train, y_train, epochs=None, batch_size=None, verbose=False):
        if epochs is None:
            epochs = config.DPSGD_N_EPOCHS
        if batch_size is None:
            batch_size = config.DPSGD_BATCH_SIZE

        # Encode labels to integers
        unique_labels = sorted(y_train.unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.inverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}

        y_encoded = y_train.map(self.label_encoder).values

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.LongTensor(y_encoded)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        input_size = X_train.shape[1]
        output_size = len(unique_labels)
        self.model = SimpleNN(input_size, config.NN_HIDDEN_LAYERS, output_size)

        # Make model compatible with Opacus
        self.model = ModuleValidator.fix(self.model)
        self.model = self.model.to(self.device)

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Calculate appropriate noise multiplier for target epsilon
        # For epsilon=0.5 (moderate privacy), use noise_multiplier ~1.3-1.5
        # Higher noise = stronger privacy = lower model accuracy
        # Formula: noise_scale = sqrt(2 * ln(1.25/delta)) / epsilon
        # For DP-SGD, noise_multiplier should be calibrated to achieve target epsilon
        # We use 1.3 as a reasonable starting point for epsilon=0.5
        noise_multiplier = 1.3 if self.epsilon <= 0.5 else 1.0

        # Attach privacy engine
        privacy_engine = PrivacyEngine()
        self.model, optimizer, train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=self.clip_norm,
        )

        if verbose:
            print(f'DP-SGD Configuration: epsilon_target={self.epsilon}, delta={self.delta}')
            print(f'Noise multiplier: {noise_multiplier}, Clip norm: {self.clip_norm}')

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Get privacy spent
            epsilon_spent = privacy_engine.get_epsilon(delta=self.delta)

            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, epsilon_spent: {epsilon_spent:.2f}/{self.epsilon}')

            # Stop if we've exceeded our privacy budget
            if epsilon_spent > self.epsilon:
                if verbose:
                    print(f'Privacy budget exceeded at epoch {epoch+1}. Stopping training.')
                    print(f'Final: epsilon_spent={epsilon_spent:.2f} > epsilon_target={self.epsilon}')
                break

        # Final privacy accounting
        final_epsilon = privacy_engine.get_epsilon(delta=self.delta)
        if verbose:
            print(f'Training completed. Final DP guarantee: (epsilon={final_epsilon:.2f}, delta={self.delta})')
            print(f'Privacy budget: {"EXCEEDED" if final_epsilon > self.epsilon else "WITHIN BUDGET"}')

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        # Decode predictions back to original labels
        predictions = predicted.cpu().numpy()
        decoded_predictions = [self.inverse_label_encoder[idx] for idx in predictions]

        return np.array(decoded_predictions)

# conduct utility assessment
def assess_utility(census_original, census_protected):
    metrics = {}

    # mean Absolute Error for income
    mae_income = mean_absolute_error(
        census_original['income'],
        census_protected['income_protected'].apply(
            lambda x: 25000 if x==config.INCOME_LABELS['low']
                   else 75000 if x==config.INCOME_LABELS['medium']
                   else 125000 if x==config.INCOME_LABELS['high']
                   else 175000
        )
    )
    metrics['income_mae'] = mae_income

    # total Variation Distance
    def total_variation_distance(orig, prot):
        orig_dist = orig.value_counts(normalize=True).sort_index()
        prot_dist = prot.value_counts(normalize=True).sort_index()
        all_cats = sorted(set(orig_dist.index) | set(prot_dist.index))
        orig_aligned = orig_dist.reindex(all_cats, fill_value=0)
        prot_aligned = prot_dist.reindex(all_cats, fill_value=0)
        return 0.5 * np.sum(np.abs(orig_aligned - prot_aligned))

    metrics['education_tvd'] = total_variation_distance(
        census_original['education'],
        census_protected['education_protected']
    )
    metrics['occupation_tvd'] = total_variation_distance(
        census_original['occupation'],
        census_protected['occupation_protected']
    )

    # Jensen-Shannon Divergence
    def calculate_jsd(orig, prot):
        orig_dist = orig.value_counts(normalize=True).sort_index()
        prot_dist = prot.value_counts(normalize=True).sort_index()
        all_cats = sorted(set(orig_dist.index) | set(prot_dist.index))
        orig_aligned = orig_dist.reindex(all_cats, fill_value=1e-10)
        prot_aligned = prot_dist.reindex(all_cats, fill_value=1e-10)
        return jensenshannon(orig_aligned, prot_aligned)

    metrics['education_jsd'] = calculate_jsd(
        census_original['education'],
        census_protected['education_protected']
    )
    metrics['occupation_jsd'] = calculate_jsd(
        census_original['occupation'],
        census_protected['occupation_protected']
    )

    # distribution preservation
    def distribution_accuracy(orig, prot):
        orig_dist = orig.value_counts(normalize=True).sort_index()
        prot_dist = prot.value_counts(normalize=True).sort_index()
        all_cats = sorted(set(orig_dist.index) | set(prot_dist.index))
        orig_aligned = orig_dist.reindex(all_cats, fill_value=0)
        prot_aligned = prot_dist.reindex(all_cats, fill_value=0)
        mae = np.mean(np.abs(orig_aligned - prot_aligned))
        return 1 - mae

    metrics['education_dist_acc'] = distribution_accuracy(
        census_original['education'],
        census_protected['education_protected']
    )
    metrics['occupation_dist_acc'] = distribution_accuracy(
        census_original['occupation'],
        census_protected['occupation_protected']
    )

    # overall utility score
    # normalize income MAE to 0-1 scale (assuming max acceptable error is $100k)
    income_utility = 1 - min(metrics['income_mae'] / 100000, 1.0)

    utility_components = [
        income_utility,                     # income normalized MAE
        1 - metrics['education_tvd'],       # education TVD
        1 - metrics['occupation_tvd'],      # occupation TVD
        1 - metrics['education_jsd'],       # education JSD
        1 - metrics['occupation_jsd'],      # occupation JSD
        metrics['education_dist_acc'],      # education distribution accuracy
        metrics['occupation_dist_acc']      # occupation distribution accuracy
    ]
    metrics['overall_utility'] = np.mean(utility_components)
    metrics['income_utility'] = income_utility

    return metrics
