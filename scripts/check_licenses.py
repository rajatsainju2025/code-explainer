#!/usr/bin/env python3
"""
License Compliance Checker

Checks for banned licenses and generates license compliance reports.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


# Banned licenses that are not allowed in the project
BANNED_LICENSES = {
    'GPL-2.0', 'GPL-2.0+', 'GPL-3.0', 'GPL-3.0+', 'GPL-3.0-only', 'GPL-3.0-or-later',
    'AGPL-3.0', 'AGPL-3.0+', 'AGPL-3.0-only', 'AGPL-3.0-or-later',
    'LGPL-2.0', 'LGPL-2.0+', 'LGPL-2.1', 'LGPL-2.1+', 'LGPL-3.0', 'LGPL-3.0+',
    'LGPL-3.0-only', 'LGPL-3.0-or-later',
    'MS-PL', 'MS-RL',  # Microsoft licenses
    'BSD-4-Clause', 'BSD-Protection',  # Problematic BSD variants
    'CC-BY-NC-SA-4.0', 'CC-BY-NC-4.0',  # Non-commercial licenses
    'WTFPL', 'Beerware', 'Unlicense'  # Permissive licenses that may cause issues
}

# Allowed licenses
ALLOWED_LICENSES = {
    'MIT', 'MIT-License',
    'BSD-2-Clause', 'BSD-3-Clause', 'BSD-3-Clause-Clear',
    'Apache-2.0', 'Apache-2.0-License',
    'ISC', 'ISC-License',
    'CC0-1.0', 'CC0-1.0-Universal',
    'CC-BY-4.0', 'CC-BY-SA-4.0',
    'Zlib', 'Zlib-License',
    'Boost-1.0',
    'PostgreSQL', 'PostgreSQL-License',
    'Python-2.0', 'Python-2.0-License',
    'BSL-1.0', 'BSL-1.0-License'
}


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {file_path}: {e}")
        sys.exit(1)


def normalize_license(license_str: str) -> str:
    """Normalize license string for comparison."""
    if not license_str:
        return 'Unknown'

    # Remove common suffixes and normalize
    normalized = license_str.strip().upper()
    normalized = normalized.replace('-LICENSE', '').replace('_LICENSE', '')
    normalized = normalized.replace('LICENSE', '').replace('-', '').replace('_', '')

    # Handle special cases
    if 'MIT' in normalized:
        return 'MIT'
    elif 'BSD3' in normalized or 'BSD-3' in normalized:
        return 'BSD-3-Clause'
    elif 'BSD2' in normalized or 'BSD-2' in normalized:
        return 'BSD-2-Clause'
    elif 'APACHE2' in normalized or 'APACHE-2' in normalized:
        return 'Apache-2.0'
    elif 'ISC' in normalized:
        return 'ISC'
    elif 'GPL' in normalized:
        return 'GPL'
    elif 'LGPL' in normalized:
        return 'LGPL'

    return license_str.strip()


def check_license_compliance(license_report: Dict[str, Any]) -> Dict[str, Any]:
    """Check license compliance against banned and allowed licenses."""

    results = {
        'compliant': True,
        'banned_licenses': [],
        'unknown_licenses': [],
        'allowed_licenses': [],
        'summary': {
            'total_packages': 0,
            'compliant_packages': 0,
            'banned_packages': 0,
            'unknown_packages': 0
        }
    }

    for package in license_report:
        if not isinstance(package, dict):
            continue

        package_name = package.get('Name', package.get('name', 'Unknown'))
        license_str = package.get('License', package.get('license', 'Unknown'))

        results['summary']['total_packages'] += 1

        # Check for banned licenses
        if license_str in BANNED_LICENSES:
            results['compliant'] = False
            results['banned_licenses'].append({
                'package': package_name,
                'license': license_str,
                'reason': 'Banned license'
            })
            results['summary']['banned_packages'] += 1
        # Check for unknown licenses
        elif license_str not in ALLOWED_LICENSES and license_str != 'Unknown':
            normalized = normalize_license(license_str)
            if normalized not in ALLOWED_LICENSES and normalized not in BANNED_LICENSES:
                results['unknown_licenses'].append({
                    'package': package_name,
                    'license': license_str,
                    'normalized': normalized
                })
                results['summary']['unknown_packages'] += 1
            elif normalized in BANNED_LICENSES:
                results['compliant'] = False
                results['banned_licenses'].append({
                    'package': package_name,
                    'license': license_str,
                    'normalized': normalized,
                    'reason': 'Banned license (normalized)'
                })
                results['summary']['banned_packages'] += 1
            else:
                results['allowed_licenses'].append({
                    'package': package_name,
                    'license': license_str
                })
                results['summary']['compliant_packages'] += 1
        else:
            results['allowed_licenses'].append({
                'package': package_name,
                'license': license_str
            })
            results['summary']['compliant_packages'] += 1

    return results


def generate_compliance_report(results: Dict[str, Any]) -> str:
    """Generate a license compliance report."""

    report_lines = ["# License Compliance Report\n"]

    # Summary
    summary = results['summary']
    report_lines.append("## Summary\n")
    report_lines.append(f"- **Total Packages**: {summary['total_packages']}")
    report_lines.append(f"- **Compliant Packages**: {summary['compliant_packages']}")
    report_lines.append(f"- **Banned Packages**: {summary['banned_packages']}")
    report_lines.append(f"- **Unknown Packages**: {summary['unknown_packages']}")
    report_lines.append("")

    # Compliance status
    if results['compliant']:
        report_lines.append("✅ **COMPLIANT**: All packages use allowed licenses.\n")
    else:
        report_lines.append("🚨 **NON-COMPLIANT**: Banned licenses detected.\n")

    # Banned licenses
    if results['banned_licenses']:
        report_lines.append("## Banned Licenses\n")
        for item in results['banned_licenses']:
            report_lines.append(f"- 🚨 **{item['package']}**: {item['license']}")
            if 'reason' in item:
                report_lines.append(f"  - Reason: {item['reason']}")
        report_lines.append("")

    # Unknown licenses
    if results['unknown_licenses']:
        report_lines.append("## Unknown Licenses (Require Review)\n")
        for item in results['unknown_licenses']:
            report_lines.append(f"- ❓ **{item['package']}**: {item['license']} (normalized: {item['normalized']})")
        report_lines.append("")

    # Allowed licenses
    if results['allowed_licenses']:
        report_lines.append("## Allowed Licenses\n")
        license_counts = {}
        for item in results['allowed_licenses']:
            license_counts[item['license']] = license_counts.get(item['license'], 0) + 1

        for license_name, count in sorted(license_counts.items()):
            report_lines.append(f"- ✅ **{license_name}**: {count} packages")
        report_lines.append("")

    # Recommendations
    report_lines.append("## Recommendations\n")

    if results['banned_licenses']:
        report_lines.append("### Immediate Actions Required")
        report_lines.append("- Replace packages with banned licenses with compliant alternatives")
        report_lines.append("- Contact maintainers if no alternatives exist")
        report_lines.append("- Update dependency management to prevent future violations")
        report_lines.append("")

    if results['unknown_licenses']:
        report_lines.append("### Review Required")
        report_lines.append("- Manually review unknown licenses for compliance")
        report_lines.append("- Add compliant licenses to ALLOWED_LICENSES if appropriate")
        report_lines.append("- Document license review decisions")
        report_lines.append("")

    report_lines.append("### General Practices")
    report_lines.append("- Regularly audit dependencies for license compliance")
    report_lines.append("- Use tools like pip-licenses for automated checking")
    report_lines.append("- Prefer MIT, BSD, and Apache 2.0 licensed packages")
    report_lines.append("- Document license choices in project governance")
    report_lines.append("")

    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Check license compliance")
    parser.add_argument("--report", required=True, help="Path to pip-licenses JSON report")
    parser.add_argument("--fail-on-banned", action="store_true",
                       help="Exit with error code if banned licenses are found")

    args = parser.parse_args()

    # Load license report
    license_report = load_json_file(args.report)

    # Check compliance
    results = check_license_compliance(license_report)

    # Print results
    print(f"License compliance check results:")
    print(f"- Total packages: {results['summary']['total_packages']}")
    print(f"- Compliant: {results['summary']['compliant_packages']}")
    print(f"- Banned: {results['summary']['banned_packages']}")
    print(f"- Unknown: {results['summary']['unknown_packages']}")

    # Generate report
    report = generate_compliance_report(results)
    print("\n" + "="*50)
    print(report)

    # Exit with error if banned licenses found and flag is set
    if args.fail_on_banned and results['banned_licenses']:
        print("Banned licenses detected! Failing build.")
        sys.exit(1)

    if not results['compliant']:
        print("License compliance issues detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()