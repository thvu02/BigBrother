import pandas as pd
import numpy as np

np.random.seed(42)

census_df = pd.read_csv('data/synthetic_census_data.csv')
census_df['occupation'] = census_df['occupation'].fillna('Not Employed')

# create enhanced ad dataset with a strong interest to demographic correlation (80%)
def create_correlated_ads(census):
    # define interest to demographic mappings
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

        # add income-based interests (80% correlation)
        if np.random.random() < 0.8:
            if person['income'] < 50000:
                interests.extend(interest_maps['income_low'])
            elif person['income'] < 100000:
                interests.extend(interest_maps['income_med'])
            else:
                interests.extend(interest_maps['income_high'])

        # add education-based interests (80% correlation)
        if np.random.random() < 0.8:
            if person['education'] in ['Less than High School', 'High School']:
                interests.extend(interest_maps['edu_low'])
            elif person['education'] in ['Some College', 'Associate Degree']:
                interests.extend(interest_maps['edu_med'])
            else:
                interests.extend(interest_maps['edu_high'])

        # add occupation-based interests (80% correlation)
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

        # create 5-7 ads per user
        for _ in range(np.random.randint(5, 8)):
            n_int = min(6, len(interests)) if interests else 3
            if interests:
                selected = list(np.random.choice(interests, n_int, replace=False))
            else:
                all_interests_flat = [i for sublist in interest_maps.values() for i in sublist]
                selected = list(np.random.choice(all_interests_flat, 3, replace=False))

            age = person['age']
            age_group = '18-24' if age<25 else '25-34' if age<35 else '35-44' if age<45 else '45-54' if age<55 else '55-64' if age<65 else '65+'
            zip_7digit = int(str(person['zip_code']) + str(np.random.randint(0, 100)).zfill(2))

            ads.append({
                'user_id': person['person_id'],
                'ad_interests': ','.join(selected),
                'target_ages': age_group,
                'target_gender': person['gender'],
                'target_locations': zip_7digit
            })

    return pd.DataFrame(ads)

# create enhanced ad dataset
ad_enhanced = create_correlated_ads(census_df)

# create enhanced census dataset with additional quasi-identifiers
census_enhanced = census_df.copy()

# create 7-digit zipcode from 5-digit one
census_enhanced['zip_code_enhanced'] = census_enhanced.apply(
    lambda row: int(str(row['zip_code']) + str(np.random.randint(0, 100)).zfill(2)), axis=1
)

# classify ages into distinct groups
census_enhanced['age_group'] = census_enhanced['age'].apply(
    lambda a: '18-24' if a<25 else '25-34' if a<35 else '35-44' if a<45 else '45-54' if a<55 else '55-64' if a<65 else '65+'
)

print(f'Census records: {len(census_enhanced):,}')
print(f'Ad records: {len(ad_enhanced):,}')
print(f'Ads per user: {len(ad_enhanced)/len(census_enhanced):.1f}')

# save enhanced datasets to csv files
census_output = 'data/enhanced_census_data.csv'
ad_output = 'data/enhanced_ad_data.csv'
census_enhanced.to_csv(census_output, index=False)
ad_enhanced.to_csv(ad_output, index=False)

# count unique interests
all_interests = set()
for interests_str in ad_enhanced['ad_interests']:
    all_interests.update(interests_str.split(','))
all_interests = sorted(list(all_interests))

print(f'\nAd Interest Statistics:')
print(f'  Unique interests: {len(all_interests)}')
print(f'  Interest list: {", ".join(all_interests[:10])}...')

# census demographics
print(f'\nCensus Demographics:')
print(f'  Age range: {census_enhanced["age"].min()} - {census_enhanced["age"].max()}')
print(f'  Income range: ${census_enhanced["income"].min():,} - ${census_enhanced["income"].max():,}')
print(f'  Education levels: {census_enhanced["education"].nunique()}')
print(f'  Occupations: {census_enhanced["occupation"].nunique()}')

# quasi-identifiers
print(f'\nQuasi-Identifiers:')
print(f'  ZIP codes (7-digit): {census_enhanced["zip_code_enhanced"].nunique():,}')
print(f'  Age groups: {census_enhanced["age_group"].nunique()}')
print(f'  Gender: {census_enhanced["gender"].nunique()}')
