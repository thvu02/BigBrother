"""
Demonstration of a simple record linkage / reidentification attack
on the provided synthetic datasets. This script matches ad records
to census records using common quasi-identifiers (age group, gender,
zip prefix).

Steps:
1. Load synthetic census and ad datasets.
2. Normalize and prepare comparable fields (age group, gender, zip prefix). 
	- load_and_prepare(census_path, ads_path)
3. For each ad record, find candidate census records that match using common quasi-identifiers. 
 	- run_linkage(census, ads)
4. Summarize and output results, create reid csv (reid_matches.csv). 
	- summarize_results(df_matches) 

Outputs:
- prints summary statistics about how many ad rows can be uniquely
  linked to a census `person_id`.
- writes `reid_matches.csv` with candidate matches and match status.
"""

from pathlib import Path
import pandas as pd
from typing import List


def age_to_age_group(age: int) -> str:
    """Convert age to age group string."""
    if age < 18:
        return "<18"
    if 18 <= age <= 24:
        return "18-24"
    if 25 <= age <= 34:
        return "25-34"
    if 35 <= age <= 44:
        return "35-44"
    if 45 <= age <= 54:
        return "45-54"
    if 55 <= age <= 64:
        return "55-64"
    return "65+"


def get_zip_prefix(x):
    """Extract first 3 digits of zip code."""
    return str(x).strip()[:3]


def load_and_prepare(census_path: Path, ads_path: Path):
	census = pd.read_csv(census_path, dtype=str)
	ads = pd.read_csv(ads_path, dtype=str)

	# Normalize and create comparable fields
	census["age"] = census["age"].astype(int)
	census["age_group"] = census["age"].apply(age_to_age_group)
	census["zip_prefix"] = census["zip_code"].apply(get_zip_prefix)

	ads["age_group"] = ads["age_group"].str.strip()
	ads["zip_prefix"] = ads["target_locations"].apply(get_zip_prefix)

	return census, ads


def run_linkage(census: pd.DataFrame, ads: pd.DataFrame):
	results = []

	for _, ad in ads.iterrows():
		candidates = census[
			(census["age_group"] == ad["age_group"]) &
			(census["gender"] == ad["gender"]) &
			(census["zip_prefix"] == ad["zip_prefix"])
		]

		candidate_ids: List[str] = candidates["person_id"].tolist()
		match_type = "none"
		if len(candidate_ids) == 1:
			match_type = "unique"
		elif len(candidate_ids) > 1:
			match_type = "ambiguous"

		# reid csv attributes
		results.append({
			"ad_id": ad.get("ad_id"),
			"user_id": ad.get("user_id"),
			"ad_age_group": ad.get("age_group"),
			"ad_gender": ad.get("gender"),
			"ad_zip_prefix": ad.get("zip_prefix"),
			"candidates_count": len(candidate_ids),
			"candidate_person_ids": ",".join(candidate_ids) if candidate_ids else "",
			"match_type": match_type,
		})

	return pd.DataFrame(results)


def summarize_results(df_matches: pd.DataFrame):
	total = len(df_matches)
	unique = (df_matches["match_type"] == "unique").sum()
	ambiguous = (df_matches["match_type"] == "ambiguous").sum()
	none = (df_matches["match_type"] == "none").sum()

	print("Linkage summary:")
	print(f"  Total ad rows evaluated: {total}")
	print(f"  Unique re-identifications: {unique}")
	print(f"  Ambiguous (multiple candidates): {ambiguous}")
	print(f"  No candidates found: {none}")


def main():
	base = Path(__file__).parent.parent
	data_dir = base / "data"
	output_dir = base / "output"
	output_dir.mkdir(exist_ok=True)
	
	census_path = data_dir / "synthetic_census_data.csv"
	ads_path = data_dir / "synthetic_facebook_ad_data.csv"

	census, ads = load_and_prepare(census_path, ads_path)
	matches = run_linkage(census, ads)
	matches.to_csv(output_dir / "reid_matches.csv", index=False)
	summarize_results(matches)



if __name__ == "__main__":
	main()

