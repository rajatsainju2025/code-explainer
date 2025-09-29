"""Robustness testing for model evaluation.

This module implements comprehensive robustness testing methods to evaluate
model performance under various adversarial and challenging conditions.
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RobustnessTest:
    """Definition of a robustness test."""

    name: str
    description: str
    transform_func: Callable[[str, float], str]
    severity_levels: List[str]
    metadata: Dict[str, Any]


@dataclass
class RobustnessResult:
    """Result of a robustness test."""

    test_name: str
    original_example: Dict[str, Any]
    transformed_example: Dict[str, Any]
    original_prediction: Any
    transformed_prediction: Any
    robustness_score: float
    severity_level: str
    metadata: Dict[str, Any]


@dataclass
class RobustnessReport:
    """Comprehensive robustness testing report."""

    total_examples: int
    total_tests: int
    test_results: List[RobustnessResult]
    aggregate_scores: Dict[str, float]
    test_summaries: Dict[str, Dict[str, Any]]
    overall_robustness_score: float


class RobustnessTransform(ABC):
    """Abstract base class for robustness transformations."""

    @abstractmethod
    def transform(self, text: str, severity: float = 0.1) -> str:
        """Apply transformation to text with given severity level."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this transformation."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of this transformation."""
        pass


class TypoTransform(RobustnessTransform):
    """Introduce typos in text."""

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            random.seed(random_seed)

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Introduce random typos in text."""
        words = text.split()
        num_typos = max(1, int(len(words) * severity))

        typo_positions = random.sample(range(len(words)), min(num_typos, len(words)))

        for pos in typo_positions:
            word = words[pos]
            if len(word) > 2:
                # Random character substitution, deletion, or insertion
                typo_type = random.choice(['substitute', 'delete', 'insert'])

                if typo_type == 'substitute':
                    char_pos = random.randint(0, len(word) - 1)
                    new_char = random.choice(string.ascii_lowercase)
                    word = word[:char_pos] + new_char + word[char_pos + 1:]
                elif typo_type == 'delete':
                    char_pos = random.randint(0, len(word) - 1)
                    word = word[:char_pos] + word[char_pos + 1:]
                elif typo_type == 'insert':
                    char_pos = random.randint(0, len(word))
                    new_char = random.choice(string.ascii_lowercase)
                    word = word[:char_pos] + new_char + word[char_pos:]

                words[pos] = word

        return ' '.join(words)

    def get_name(self) -> str:
        return "typo"

    def get_description(self) -> str:
        return "Introduce random typos (substitution, deletion, insertion)"


class CaseTransform(RobustnessTransform):
    """Change text case patterns."""

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Change case of random words."""
        words = text.split()
        num_changes = max(1, int(len(words) * severity))

        change_positions = random.sample(range(len(words)), min(num_changes, len(words)))

        for pos in change_positions:
            word = words[pos]
            case_type = random.choice(['upper', 'lower', 'title', 'random'])

            if case_type == 'upper':
                words[pos] = word.upper()
            elif case_type == 'lower':
                words[pos] = word.lower()
            elif case_type == 'title':
                words[pos] = word.title()
            elif case_type == 'random':
                words[pos] = ''.join(
                    c.upper() if random.random() < 0.5 else c.lower()
                    for c in word
                )

        return ' '.join(words)

    def get_name(self) -> str:
        return "case_change"

    def get_description(self) -> str:
        return "Change case patterns in text"


class WhitespaceTransform(RobustnessTransform):
    """Modify whitespace patterns."""

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Modify whitespace in text."""
        # Add extra spaces
        if random.random() < severity:
            text = re.sub(r' ', '  ', text)

        # Add tabs and newlines
        if random.random() < severity:
            words = text.split()
            for i in range(len(words)):
                if random.random() < severity / 2:
                    words[i] += random.choice(['\t', '\n', '  '])
            text = ' '.join(words)

        # Remove some spaces
        if random.random() < severity:
            text = re.sub(r'  +', ' ', text)

        return text

    def get_name(self) -> str:
        return "whitespace"

    def get_description(self) -> str:
        return "Modify whitespace patterns"


class PunctuationTransform(RobustnessTransform):
    """Modify punctuation."""

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Modify punctuation in text."""
        # Remove some punctuation
        if random.random() < severity:
            text = re.sub(r'[,.;:]', '', text)

        # Add extra punctuation
        if random.random() < severity:
            words = text.split()
            for i in range(len(words)):
                if random.random() < severity / 2:
                    words[i] += random.choice(['.', ',', ';', '!', '?'])
            text = ' '.join(words)

        # Change punctuation
        if random.random() < severity:
            punctuation_map = {'.': '!', '!': '?', '?': '.', ',': ';', ';': ','}
            for old, new in punctuation_map.items():
                if random.random() < 0.3:
                    text = text.replace(old, new)

        return text

    def get_name(self) -> str:
        return "punctuation"

    def get_description(self) -> str:
        return "Modify punctuation patterns"


class SynonymTransform(RobustnessTransform):
    """Replace words with synonyms."""

    def __init__(self):
        # Simple synonym dictionary - in practice, you'd use WordNet or similar
        self.synonyms = {
            'function': ['method', 'procedure', 'routine'],
            'variable': ['var', 'parameter', 'argument'],
            'return': ['give_back', 'output', 'yield'],
            'create': ['make', 'build', 'generate'],
            'delete': ['remove', 'erase', 'eliminate'],
            'update': ['modify', 'change', 'alter'],
            'good': ['nice', 'excellent', 'great'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'minor']
        }

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Replace words with synonyms."""
        words = text.split()
        num_replacements = max(1, int(len(words) * severity))

        replacement_positions = []
        for i, word in enumerate(words):
            word_lower = word.lower().strip(string.punctuation)
            if word_lower in self.synonyms:
                replacement_positions.append(i)

        if replacement_positions:
            selected_positions = random.sample(
                replacement_positions,
                min(num_replacements, len(replacement_positions))
            )

            for pos in selected_positions:
                word = words[pos]
                word_lower = word.lower().strip(string.punctuation)
                if word_lower in self.synonyms:
                    synonym = random.choice(self.synonyms[word_lower])
                    # Preserve case and punctuation
                    if word.isupper():
                        synonym = synonym.upper()
                    elif word.istitle():
                        synonym = synonym.title()

                    # Preserve punctuation
                    punctuation = ''.join(c for c in word if c in string.punctuation)
                    words[pos] = synonym + punctuation

        return ' '.join(words)

    def get_name(self) -> str:
        return "synonym"

    def get_description(self) -> str:
        return "Replace words with synonyms"


class CodeObfuscationTransform(RobustnessTransform):
    """Obfuscate code while preserving functionality."""

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Apply code obfuscation."""
        # Variable name obfuscation
        if random.random() < severity:
            # Find variable names (simplified)
            var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            variables = set(re.findall(var_pattern, text))

            # Filter out keywords
            keywords = {
                'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
                'import', 'from', 'return', 'yield', 'break', 'continue', 'pass',
                'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'
            }

            variables = variables - keywords

            # Replace some variables with obfuscated names
            num_obfuscations = max(1, int(len(variables) * severity))
            vars_to_obfuscate = random.sample(list(variables), min(num_obfuscations, len(variables)))

            for var in vars_to_obfuscate:
                obfuscated_name = f"var_{random.randint(1000, 9999)}"
                text = re.sub(r'\b' + re.escape(var) + r'\b', obfuscated_name, text)

        # Add unnecessary parentheses
        if random.random() < severity:
            # Simple expression wrapping
            text = re.sub(r'(\w+\s*[+\-*/]\s*\w+)', r'(\1)', text)

        # Add extra whitespace in code
        if random.random() < severity:
            text = re.sub(r'([=+\-*/,()])', r' \1 ', text)
            text = re.sub(r'\s+', ' ', text)  # Clean up multiple spaces

        return text

    def get_name(self) -> str:
        return "code_obfuscation"

    def get_description(self) -> str:
        return "Obfuscate code while preserving functionality"


class AdversarialTextTransform(RobustnessTransform):
    """Apply adversarial text modifications."""

    def transform(self, text: str, severity: float = 0.1) -> str:
        """Apply adversarial modifications."""
        # Character substitution with visually similar characters
        substitutions = {
            'a': ['α', 'а'],  # Greek alpha, Cyrillic a
            'e': ['е'],  # Cyrillic e
            'o': ['о', '0'],  # Cyrillic o, zero
            'p': ['р'],  # Cyrillic p
            'c': ['с'],  # Cyrillic c
            'x': ['х'],  # Cyrillic x
        }

        if random.random() < severity:
            for char, alternatives in substitutions.items():
                if char in text.lower() and random.random() < 0.3:
                    replacement = random.choice(alternatives)
                    # Maintain case
                    if char.isupper():
                        replacement = replacement.upper()

                    # Replace some occurrences
                    positions = [i for i, c in enumerate(text.lower()) if c == char]
                    if positions:
                        pos = random.choice(positions)
                        text = text[:pos] + replacement + text[pos + 1:]

        # Add invisible characters
        if random.random() < severity:
            invisible_chars = ['\u200b', '\u200c', '\u200d']  # Zero-width characters
            words = text.split()
            for i in range(len(words)):
                if random.random() < severity / 2:
                    words[i] += random.choice(invisible_chars)
            text = ' '.join(words)

        return text

    def get_name(self) -> str:
        return "adversarial"

    def get_description(self) -> str:
        return "Apply adversarial text modifications"


class RobustnessTester:
    """Main robustness testing class."""

    def __init__(
        self,
        transforms: Optional[List[RobustnessTransform]] = None,
        severity_levels: Optional[List[float]] = None,
        random_seed: Optional[int] = None
    ):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.transforms = transforms or [
            TypoTransform(random_seed),
            CaseTransform(),
            WhitespaceTransform(),
            PunctuationTransform(),
            SynonymTransform(),
            CodeObfuscationTransform(),
            AdversarialTextTransform()
        ]

        self.severity_levels = severity_levels or [0.05, 0.1, 0.2, 0.3]

    def test_robustness(
        self,
        examples: List[Dict[str, Any]],
        predict_func: Callable[[Dict[str, Any]], Any],
        fields_to_test: Optional[List[str]] = None,
        max_examples_per_test: Optional[int] = None
    ) -> RobustnessReport:
        """Run comprehensive robustness tests."""
        fields_to_test = fields_to_test or ['code', 'explanation']

        # Limit examples if specified
        if max_examples_per_test:
            examples = random.sample(examples, min(max_examples_per_test, len(examples)))

        all_results = []
        test_summaries = {}

        logger.info(f"Running robustness tests on {len(examples)} examples...")

        for transform in self.transforms:
            transform_name = transform.get_name()
            logger.info(f"Running {transform_name} tests...")

            transform_results = []

            for severity in self.severity_levels:
                severity_results = []

                for example in examples:
                    for field in fields_to_test:
                        if field not in example:
                            continue

                        # Get original prediction
                        original_prediction = predict_func(example)

                        # Apply transformation
                        transformed_example = example.copy()
                        transformed_example[field] = transform.transform(
                            example[field], severity
                        )

                        # Get transformed prediction
                        transformed_prediction = predict_func(transformed_example)

                        # Calculate robustness score
                        robustness_score = self._calculate_robustness_score(
                            original_prediction, transformed_prediction
                        )

                        result = RobustnessResult(
                            test_name=transform_name,
                            original_example=example,
                            transformed_example=transformed_example,
                            original_prediction=original_prediction,
                            transformed_prediction=transformed_prediction,
                            robustness_score=robustness_score,
                            severity_level=str(severity),
                            metadata={
                                'field': field,
                                'transform_description': transform.get_description()
                            }
                        )

                        severity_results.append(result)
                        all_results.append(result)

                transform_results.extend(severity_results)

            # Calculate summary statistics for this transform
            if transform_results:
                scores = [r.robustness_score for r in transform_results]
                test_summaries[transform_name] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores)),
                    'num_tests': len(transform_results),
                    'description': transform.get_description()
                }

        # Calculate aggregate scores
        aggregate_scores = self._calculate_aggregate_scores(all_results)

        # Calculate overall robustness score
        if all_results:
            overall_score = float(np.mean([r.robustness_score for r in all_results]))
        else:
            overall_score = 0.0

        return RobustnessReport(
            total_examples=len(examples),
            total_tests=len(all_results),
            test_results=all_results,
            aggregate_scores=aggregate_scores,
            test_summaries=test_summaries,
            overall_robustness_score=overall_score
        )

    def _calculate_robustness_score(self, original_pred: Any, transformed_pred: Any) -> float:
        """Calculate robustness score between predictions."""
        # Simple implementation - you might want more sophisticated scoring
        if isinstance(original_pred, str) and isinstance(transformed_pred, str):
            # Text similarity using simple overlap
            original_words = set(original_pred.lower().split())
            transformed_words = set(transformed_pred.lower().split())

            if not original_words and not transformed_words:
                return 1.0

            intersection = original_words.intersection(transformed_words)
            union = original_words.union(transformed_words)

            return len(intersection) / len(union) if union else 1.0

        elif isinstance(original_pred, (int, float)) and isinstance(transformed_pred, (int, float)):
            # Numerical similarity
            if original_pred == 0 and transformed_pred == 0:
                return 1.0

            diff = abs(original_pred - transformed_pred)
            max_val = max(abs(original_pred), abs(transformed_pred))

            return 1.0 - (diff / max_val) if max_val > 0 else 1.0

        else:
            # Exact match for other types
            return 1.0 if original_pred == transformed_pred else 0.0

    def _calculate_aggregate_scores(self, results: List[RobustnessResult]) -> Dict[str, float]:
        """Calculate aggregate robustness scores."""
        if not results:
            return {}

        # Overall statistics
        scores = [r.robustness_score for r in results]
        aggregate = {
            'mean_robustness': float(np.mean(scores)),
            'std_robustness': float(np.std(scores)),
            'min_robustness': float(np.min(scores)),
            'max_robustness': float(np.max(scores)),
            'median_robustness': float(np.median(scores))
        }

        # By severity level
        severity_groups = {}
        for result in results:
            severity = result.severity_level
            if severity not in severity_groups:
                severity_groups[severity] = []
            severity_groups[severity].append(result.robustness_score)

        for severity, severity_scores in severity_groups.items():
            aggregate[f'mean_robustness_severity_{severity}'] = float(np.mean(severity_scores))

        # By test type
        test_groups = {}
        for result in results:
            test_name = result.test_name
            if test_name not in test_groups:
                test_groups[test_name] = []
            test_groups[test_name].append(result.robustness_score)

        for test_name, test_scores in test_groups.items():
            aggregate[f'mean_robustness_{test_name}'] = float(np.mean(test_scores))

        return aggregate

    def save_report(self, report: RobustnessReport, output_file: Union[str, Path]):
        """Save robustness report to file."""
        report_dict = {
            'total_examples': report.total_examples,
            'total_tests': report.total_tests,
            'overall_robustness_score': report.overall_robustness_score,
            'aggregate_scores': report.aggregate_scores,
            'test_summaries': report.test_summaries,
            'test_results': [
                {
                    'test_name': result.test_name,
                    'robustness_score': result.robustness_score,
                    'severity_level': result.severity_level,
                    'metadata': result.metadata,
                    'original_prediction': str(result.original_prediction),
                    'transformed_prediction': str(result.transformed_prediction)
                }
                for result in report.test_results
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Robustness report saved to {output_file}")


def run_robustness_tests(
    examples: List[Dict[str, Any]],
    predict_func: Callable[[Dict[str, Any]], Any],
    output_file: Union[str, Path],
    test_types: Optional[List[str]] = None,
    severity_levels: Optional[List[float]] = None,
    max_examples: Optional[int] = None,
    random_seed: Optional[int] = None
) -> RobustnessReport:
    """Run robustness tests with specified configuration."""
    # Create transforms based on test types
    transforms = []
    if test_types is None:
        test_types = ['typo', 'case', 'whitespace', 'punctuation', 'synonym', 'code_obfuscation', 'adversarial']

    transform_map = {
        'typo': TypoTransform(random_seed),
        'case': CaseTransform(),
        'whitespace': WhitespaceTransform(),
        'punctuation': PunctuationTransform(),
        'synonym': SynonymTransform(),
        'code_obfuscation': CodeObfuscationTransform(),
        'adversarial': AdversarialTextTransform()
    }

    for test_type in test_types:
        if test_type in transform_map:
            transforms.append(transform_map[test_type])

    tester = RobustnessTester(transforms, severity_levels or [0.05, 0.1, 0.2, 0.3], random_seed)
    report = tester.test_robustness(examples, predict_func, max_examples_per_test=max_examples)

    tester.save_report(report, output_file)
    return report
