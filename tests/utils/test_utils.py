import unittest
from unittest.mock import MagicMock, patch

from app.utils.utils import (
    ADDITIONAL_METRICS,
    BASE_CLASS_WHITELIST,
    DEFAULT_PRIMARY_METRIC,
    DISALLOWED_TOKENS,
    ENSEMBLE_CLASS_NAMES,
    parse_algo_class,
    with_retry,
)


class TestConstants(unittest.TestCase):
    """Test that all constants have expected values and types."""

    def test_default_primary_metric(self):
        """Test DEFAULT_PRIMARY_METRIC constant."""
        self.assertEqual(DEFAULT_PRIMARY_METRIC, "area_under_roc_curve")
        self.assertIsInstance(DEFAULT_PRIMARY_METRIC, str)

    def test_additional_metrics(self):
        """Test ADDITIONAL_METRICS constant."""
        expected_metrics = [
            "predictive_accuracy",
        ]
        self.assertEqual(ADDITIONAL_METRICS, expected_metrics)
        self.assertIsInstance(ADDITIONAL_METRICS, list)
        for metric in ADDITIONAL_METRICS:
            self.assertIsInstance(metric, str)

    def test_base_class_whitelist(self):
        """Test BASE_CLASS_WHITELIST constant."""
        expected_classes = {
            "LogisticRegression",
            "RidgeClassifier",
            "LinearSVC",
            "SVC",
            "KNeighborsClassifier",
            "NearestCentroid",
            "GaussianNB",
            "MultinomialNB",
            "BernoulliNB",
            "DecisionTreeClassifier",
            "ExtraTreeClassifier",
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "AdaBoostClassifier",
            "MLPClassifier",
            "QuadraticDiscriminantAnalysis",
            "LinearDiscriminantAnalysis",
        }
        self.assertEqual(BASE_CLASS_WHITELIST, expected_classes)
        self.assertIsInstance(BASE_CLASS_WHITELIST, set)
        for class_name in BASE_CLASS_WHITELIST:
            self.assertIsInstance(class_name, str)

    def test_disallowed_tokens(self):
        """Test DISALLOWED_TOKENS constant."""
        expected_tokens = [
            "Pipeline",
            "FeatureUnion",
            "ColumnTransformer",
            "SearchCV",
            "CalibratedClassifier",
            "Stacking",
            "Bagging",
            "Voting",
        ]
        self.assertEqual(DISALLOWED_TOKENS, expected_tokens)
        self.assertIsInstance(DISALLOWED_TOKENS, list)
        for token in DISALLOWED_TOKENS:
            self.assertIsInstance(token, str)

    def test_ensemble_class_names(self):
        """Test ENSEMBLE_CLASS_NAMES constant."""
        expected_ensemble_classes = {
            "RandomForestClassifier",
            "ExtraTreesClassifier",
            "GradientBoostingClassifier",
            "HistGradientBoostingClassifier",
            "AdaBoostClassifier",
        }
        self.assertEqual(ENSEMBLE_CLASS_NAMES, expected_ensemble_classes)
        self.assertIsInstance(ENSEMBLE_CLASS_NAMES, set)
        for class_name in ENSEMBLE_CLASS_NAMES:
            self.assertIsInstance(class_name, str)

    def test_ensemble_classes_are_subset_of_base_classes(self):
        """Test that all ensemble classes are in the base class whitelist."""
        self.assertTrue(ENSEMBLE_CLASS_NAMES.issubset(BASE_CLASS_WHITELIST))


class TestParseAlgoClass(unittest.TestCase):
    """Test the parse_algo_class function."""

    def test_parse_simple_class_name(self):
        """Test parsing a simple class name."""
        result = parse_algo_class("LogisticRegression")
        self.assertEqual(result, "LogisticRegression")

    def test_parse_module_dotted_class_name(self):
        """Test parsing a class name with module path."""
        result = parse_algo_class("sklearn.linear_model.LogisticRegression")
        self.assertEqual(result, "LogisticRegression")

    def test_parse_nested_module_class_name(self):
        """Test parsing a class name with deeply nested module path."""
        result = parse_algo_class("sklearn.ensemble.forest.RandomForestClassifier")
        self.assertEqual(result, "RandomForestClassifier")

    def test_parse_class_with_parameters(self):
        """Test parsing a class name followed by parameters."""
        result = parse_algo_class("sklearn.svm.SVC C=1.0")
        self.assertEqual(result, "SVC")

    def test_parse_class_with_parameters_in_braces(self):
        """Test parsing a class name followed by parameters in braces."""
        result = parse_algo_class("sklearn.svm.SVC(8)")
        self.assertEqual(result, "SVC")

    def test_parse_class_with_complex_parameters(self):
        """Test parsing a class name with complex parameter string."""
        result = parse_algo_class(
            "sklearn.ensemble.RandomForestClassifier n_estimators=100 max_depth=None"
        )
        self.assertEqual(result, "RandomForestClassifier")

    def test_parse_empty_string(self):
        """Test parsing an empty string."""
        result = parse_algo_class("")
        self.assertEqual(result, "")

    def test_parse_single_dot(self):
        """Test parsing a single dot."""
        result = parse_algo_class(".")
        self.assertEqual(result, "")

    def test_parse_trailing_dot(self):
        """Test parsing a string with trailing dot."""
        result = parse_algo_class("sklearn.svm.")
        self.assertEqual(result, "")

    def test_parse_multiple_spaces_in_parameters(self):
        """Test parsing class name with multiple spaces in parameters."""
        result = parse_algo_class(
            "sklearn.tree.DecisionTreeClassifier   max_depth=5   random_state=42"
        )
        self.assertEqual(result, "DecisionTreeClassifier")


class TestWithRetry(unittest.TestCase):
    """Test the with_retry function."""

    def test_successful_function_call(self):
        """Test that successful function calls return immediately."""
        mock_fn = MagicMock(return_value="success")
        result = with_retry(mock_fn)

        self.assertEqual(result, "success")
        mock_fn.assert_called_once()

    def test_function_fails_once_then_succeeds(self):
        """Test that function is retried when it fails initially."""
        mock_fn = MagicMock(side_effect=[Exception("first failure"), "success"])

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = with_retry(mock_fn, retries=2)

        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 2)

    def test_function_fails_all_retries(self):
        """Test that function raises exception after all retries are exhausted."""
        mock_fn = MagicMock(side_effect=Exception("persistent failure"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with self.assertRaises(Exception) as context:
                with_retry(mock_fn, retries=2)

        self.assertEqual(str(context.exception), "persistent failure")
        self.assertEqual(mock_fn.call_count, 3)  # initial + 2 retries

    def test_custom_retry_count(self):
        """Test that custom retry count is respected."""
        mock_fn = MagicMock(side_effect=Exception("failure"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with self.assertRaises(Exception):
                with_retry(mock_fn, retries=5)

        self.assertEqual(mock_fn.call_count, 6)  # initial + 5 retries

    @patch("time.sleep")
    @patch("random.uniform")
    def test_exponential_backoff_sleep_calculation(self, mock_uniform, mock_sleep):
        """Test that sleep time follows exponential backoff pattern."""
        mock_uniform.return_value = 0.1  # Fixed random component
        mock_fn = MagicMock(
            side_effect=[Exception("fail1"), Exception("fail2"), "success"]
        )

        result = with_retry(mock_fn, retries=3, base_sleep=2.0)

        self.assertEqual(result, "success")

        # Check sleep was called with exponentially increasing times
        expected_calls = [
            unittest.mock.call(2.0 * (2**0) + 0.1),  # First retry: 2.0 * 1 + 0.1 = 2.1
            unittest.mock.call(2.0 * (2**1) + 0.1),  # Second retry: 2.0 * 2 + 0.1 = 4.1
        ]
        mock_sleep.assert_has_calls(expected_calls)

    def test_logging_success_case(self):
        """Test that successful calls don't log anything."""
        mock_logger = MagicMock()
        mock_fn = MagicMock(return_value="success")

        result = with_retry(mock_fn, logger=mock_logger)

        self.assertEqual(result, "success")
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_logging_warning_on_retry(self):
        """Test that warnings are logged during retries."""
        mock_logger = MagicMock()
        mock_fn = MagicMock(side_effect=[Exception("retry failure"), "success"])

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = with_retry(mock_fn, retries=2, logger=mock_logger)

        self.assertEqual(result, "success")
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        self.assertIn("retry failure", warning_call)
        self.assertIn("retrying in", warning_call)

    def test_logging_error_on_final_failure(self):
        """Test that error is logged when all retries are exhausted."""
        mock_logger = MagicMock()
        mock_fn = MagicMock(side_effect=Exception("final failure"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with self.assertRaises(Exception):
                with_retry(mock_fn, retries=1, logger=mock_logger)

        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        self.assertIn("final failure", error_call)
        self.assertIn("failed after 1 retries", error_call)

    def test_no_logger_provided(self):
        """Test that function works correctly when no logger is provided."""
        mock_fn = MagicMock(side_effect=[Exception("failure"), "success"])

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = with_retry(mock_fn, retries=2, logger=None)

        self.assertEqual(result, "success")

    @patch("random.uniform")
    def test_random_jitter_range(self, mock_uniform):
        """Test that random jitter is within expected range."""
        mock_uniform.return_value = 0.25  # Maximum jitter
        mock_fn = MagicMock(side_effect=[Exception("fail"), "success"])

        with patch("time.sleep") as mock_sleep:
            with_retry(mock_fn, retries=2, base_sleep=1.0)

        # Verify uniform was called with correct range
        mock_uniform.assert_called_with(0, 0.25)

        # Verify sleep was called with base_sleep + jitter
        mock_sleep.assert_called_once_with(1.0 * (2**0) + 0.25)  # 1.25

    def test_zero_retries(self):
        """Test behavior when retries is set to 0."""
        mock_fn = MagicMock(side_effect=Exception("immediate failure"))

        with self.assertRaises(Exception) as context:
            with_retry(mock_fn, retries=0)

        self.assertEqual(str(context.exception), "immediate failure")
        mock_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
