# Dataset Governance and Intake

Standardize data contributions and ensure compliance.

Intake YAML (see `data/INTAKE_TEMPLATE.yaml`)
- id, title, description
- source_url, license, license_url
- added_by, added_at (UTC)
- citation, notes

Checks (CI)
- Validate intake YAML on PRs
- Run PII/secrets scanners on modified data files
- Ensure train/test split rules and contamination checks

Contributor checklist
- You have rights to share the data
- License permits use/distribution
- Remove PII/secrets and add redaction notes if needed
