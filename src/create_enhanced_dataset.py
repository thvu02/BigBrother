"""
================================================================================
ENHANCED DATASET GENERATOR
================================================================================

Creates enhanced census and ad datasets with strong demographic-interest
correlations for privacy attack analysis.

Team: Brandon Diep, Chidiebere Okpara, Thi Thuy Trang Tran, Trung Vu
Course: CS5510 - Privacy and Security

Output Files:
- data/enhanced_census_data.csv
- data/enhanced_ad_data.csv

Strategy:
- Creates 80% correlation between demographics and ad interests
- Generates 5-7 ads per user
- Adds enhanced quasi-identifiers (7-digit ZIP, age groups)

Run this script once to generate the enhanced dataset, then use
master_privacy_analysis.py which will load the pre-generated data.
================================================================================
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

print('='*80)
print('ENHANCED DATASET GENERATOR')
print('='*80)

# =============================================================================
# LOAD ORIGINAL DATA
# =============================================================================
print('\nLoading original data...')
census_df = pd.read_csv('data/synthetic_census_data.csv')
census_df['occupation'] = census_df['occupation'].fillna('Not Employed')

print(f'Original Data:')
print(f'  Census records: {len(census_df):,}')
print(f'  Original ad records: Not used (will create enhanced ads)')

# =============================================================================
# ENHANCED DATASET CREATION
# =============================================================================
print('\n' + '='*80)
print('CREATING ENHANCED DATASET')
print('Creating dataset with strong demographic-interest correlations (80%)')
print('='*80)

def create_correlated_ads(census):
    """
    Create ad dataset with strong interest-demographic correlations

    Strategy:
    - Map demographics to specific interests (80% correlation)
    - Income: Low/Med/High -> Different interest sets
    - Education: Low/Med/High -> Different interest sets
    - Occupation: Tech/Health/Business/Other -> Different interest sets
    - Generate 5-7 ads per user with correlated interests

    Returns:
        DataFrame with user_id, ad_interests, target demographics
    """
    # Define interest-demographic mappings
    interest_maps = {
        'income_low': ['coupons', 'discount_shopping', 'thrift_stores', 'fast_food'],
        'income_med': ['online_shopping', 'streaming', 'chain_restaurants'],
        'income_high': ['luxury_travel', 'fine_dining', 'real_estate', 'investment', 'golf'],
        'edu_low': ['job_training', 'vocational_schools', 'ged_programs'],
        'edu_med': ['community_college', 'online_courses'],
        'edu_high': ['graduate_schools', 'academic_journals', 'research'],
        'occ_tech': ['programming', 'ai_ml', 'tech_conferences', 'github'],
        'occ_health': ['medical_journals', 'healthcare_tech'],
        'occ_biz': ['business_strategy', 'leadership', 'mba_programs'],
        'occ_other': ['career_advice', 'job_search']
    }

    ads = []
    for _, person in census.iterrows():
        interests = []

        # Add income-based interests (80% correlation)
        if np.random.random() < 0.8:
            if person['income'] < 50000:
                interests.extend(interest_maps['income_low'])
            elif person['income'] < 100000:
                interests.extend(interest_maps['income_med'])
            else:
                interests.extend(interest_maps['income_high'])

        # Add education-based interests (80% correlation)
        if np.random.random() < 0.8:
            if person['education'] in ['Less than High School', 'High School']:
                interests.extend(interest_maps['edu_low'])
            elif person['education'] in ['Some College', 'Associate Degree']:
                interests.extend(interest_maps['edu_med'])
            else:
                interests.extend(interest_maps['edu_high'])

        # Add occupation-based interests (80% correlation)
        if np.random.random() < 0.8:
            if person['occupation'] == 'Computer and Mathematical':
                interests.extend(interest_maps['occ_tech'])
            elif person['occupation'] == 'Healthcare':
                interests.extend(interest_maps['occ_health'])
            elif person['occupation'] in ['Management', 'Business and Financial']:
                interests.extend(interest_maps['occ_biz'])
            else:
                interests.extend(interest_maps['occ_other'])

        interests = list(set(interests))

        # Generate 5-7 ads per user
        for _ in range(np.random.randint(5, 8)):
            n_int = min(6, len(interests)) if interests else 3
            if interests:
                selected = list(np.random.choice(interests, n_int, replace=False))
            else:
                all_interests_flat = [i for sublist in interest_maps.values() for i in sublist]
                selected = list(np.random.choice(all_interests_flat, 3, replace=False))

            age = person['age']
            age_group = '18-24' if age<25 else '25-34' if age<35 else '35-44' if age<45 else '45-54' if age<55 else '55-64' if age<65 else '65+'
            zip_5digit = int(str(person['zip_code']) + str(np.random.randint(0, 100)).zfill(2))

            ads.append({
                'user_id': person['person_id'],
                'ad_interests': ','.join(selected),
                'target_ages': age_group,
                'target_gender': person['gender'],
                'target_locations': zip_5digit
            })

    return pd.DataFrame(ads)

# Create enhanced ad dataset
print('\nGenerating enhanced ad dataset with correlations...')
ad_enhanced = create_correlated_ads(census_df)

# Create enhanced census dataset with additional quasi-identifiers
print('Enhancing census dataset with quasi-identifiers...')
census_enhanced = census_df.copy()
census_enhanced['zip_code_enhanced'] = census_enhanced.apply(
    lambda row: int(str(row['zip_code']) + str(np.random.randint(0, 100)).zfill(2)), axis=1
)
census_enhanced['age_group'] = census_enhanced['age'].apply(
    lambda a: '18-24' if a<25 else '25-34' if a<35 else '35-44' if a<45 else '45-54' if a<55 else '55-64' if a<65 else '65+'
)

print(f'\nEnhanced Dataset Created:')
print(f'  Census records: {len(census_enhanced):,}')
print(f'  Ad records: {len(ad_enhanced):,}')
print(f'  Ads per user: {len(ad_enhanced)/len(census_enhanced):.1f}')

# =============================================================================
# SAVE TO CSV
# =============================================================================
print('\n' + '='*80)
print('SAVING ENHANCED DATASETS TO CSV')
print('='*80)

census_output = 'data/enhanced_census_data.csv'
ad_output = 'data/enhanced_ad_data.csv'

census_enhanced.to_csv(census_output, index=False)
print(f'\nSaved enhanced census to: {census_output}')
print(f'  Columns: {list(census_enhanced.columns)}')
print(f'  Records: {len(census_enhanced):,}')

ad_enhanced.to_csv(ad_output, index=False)
print(f'\nSaved enhanced ads to: {ad_output}')
print(f'  Columns: {list(ad_enhanced.columns)}')
print(f'  Records: {len(ad_enhanced):,}')

# =============================================================================
# DATASET STATISTICS
# =============================================================================
print('\n' + '='*80)
print('DATASET STATISTICS')
print('='*80)

# Count unique interests
all_interests = set()
for interests_str in ad_enhanced['ad_interests']:
    all_interests.update(interests_str.split(','))
all_interests = sorted(list(all_interests))

print(f'\nAd Interest Statistics:')
print(f'  Unique interests: {len(all_interests)}')
print(f'  Interest list: {", ".join(all_interests[:10])}...')

# Census demographics
print(f'\nCensus Demographics:')
print(f'  Age range: {census_enhanced["age"].min()} - {census_enhanced["age"].max()}')
print(f'  Income range: ${census_enhanced["income"].min():,} - ${census_enhanced["income"].max():,}')
print(f'  Education levels: {census_enhanced["education"].nunique()}')
print(f'  Occupations: {census_enhanced["occupation"].nunique()}')

# Quasi-identifiers
print(f'\nQuasi-Identifiers:')
print(f'  ZIP codes (7-digit): {census_enhanced["zip_code_enhanced"].nunique():,}')
print(f'  Age groups: {census_enhanced["age_group"].nunique()}')
print(f'  Gender: {census_enhanced["gender"].nunique()}')

print('\n' + '='*80)
print('ENHANCED DATASET GENERATION COMPLETE')
print('='*80)
print('\nNext Steps:')
print('1. Enhanced datasets saved to data/ directory')
print('2. Run master_privacy_analysis.py to perform analysis')
print('   (Script will automatically load enhanced datasets)')
print('='*80)
