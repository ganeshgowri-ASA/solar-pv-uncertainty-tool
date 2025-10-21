"""
Data handling and validation module for PV uncertainty analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import io


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None


class PVDataValidator:
    """
    Validator for PV measurement data.
    """

    @staticmethod
    def validate_power_data(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate PV power measurement data.

        Args:
            df: DataFrame with measurement data
            required_columns: List of required column names

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []

        if required_columns is None:
            required_columns = ['irradiance', 'temperature', 'power']

        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            return ValidationResult(False, errors, warnings)

        # Create a copy for cleaning
        cleaned_df = df.copy()

        # Check for negative values
        for col in required_columns:
            if col in cleaned_df.columns:
                negative_mask = cleaned_df[col] < 0
                if negative_mask.any():
                    n_negative = negative_mask.sum()
                    warnings.append(
                        f"Column '{col}' has {n_negative} negative values. "
                        f"Setting to NaN."
                    )
                    cleaned_df.loc[negative_mask, col] = np.nan

        # Check for NaN values
        for col in required_columns:
            if col in cleaned_df.columns:
                nan_mask = cleaned_df[col].isna()
                if nan_mask.any():
                    n_nan = nan_mask.sum()
                    warnings.append(
                        f"Column '{col}' has {n_nan} NaN values "
                        f"({n_nan/len(cleaned_df)*100:.1f}%)."
                    )

        # Check for unrealistic values
        if 'irradiance' in cleaned_df.columns:
            high_irrad = cleaned_df['irradiance'] > 1500
            if high_irrad.any():
                warnings.append(
                    f"Irradiance values > 1500 W/m² detected "
                    f"({high_irrad.sum()} records). Check data quality."
                )

        if 'temperature' in cleaned_df.columns:
            low_temp = cleaned_df['temperature'] < -40
            high_temp = cleaned_df['temperature'] > 100
            if low_temp.any() or high_temp.any():
                warnings.append(
                    f"Temperature values outside typical range (-40°C to 100°C) detected."
                )

        # Check data length
        if len(cleaned_df) < 10:
            warnings.append(
                f"Dataset has only {len(cleaned_df)} records. "
                f"Statistical analysis may be unreliable."
            )

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, errors, warnings, cleaned_df)

    @staticmethod
    def validate_uncertainty_inputs(
        irradiance: float,
        irradiance_unc: float,
        temperature: float,
        temperature_unc: float,
        power: float,
        power_unc: float
    ) -> ValidationResult:
        """
        Validate uncertainty input parameters.

        Args:
            irradiance: Irradiance value
            irradiance_unc: Irradiance uncertainty
            temperature: Temperature value
            temperature_unc: Temperature uncertainty
            power: Power value
            power_unc: Power uncertainty

        Returns:
            ValidationResult object
        """
        errors = []
        warnings = []

        # Check for negative values
        if irradiance < 0:
            errors.append("Irradiance cannot be negative")
        if power < 0:
            errors.append("Power cannot be negative")

        # Check for unrealistic values
        if irradiance > 1500:
            warnings.append("Irradiance > 1500 W/m² is unusually high")
        if temperature < -40 or temperature > 100:
            warnings.append("Temperature outside typical range (-40°C to 100°C)")

        # Check uncertainty values
        if irradiance_unc < 0:
            errors.append("Irradiance uncertainty cannot be negative")
        if temperature_unc < 0:
            errors.append("Temperature uncertainty cannot be negative")
        if power_unc < 0:
            errors.append("Power uncertainty cannot be negative")

        # Check relative uncertainties
        if irradiance > 0:
            rel_unc_irrad = irradiance_unc / irradiance * 100
            if rel_unc_irrad > 50:
                warnings.append(
                    f"Irradiance relative uncertainty is very high ({rel_unc_irrad:.1f}%)"
                )

        if power > 0:
            rel_unc_power = power_unc / power * 100
            if rel_unc_power > 50:
                warnings.append(
                    f"Power relative uncertainty is very high ({rel_unc_power:.1f}%)"
                )

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, errors, warnings)


class PVDataHandler:
    """
    Handler for importing and exporting PV measurement data.
    """

    @staticmethod
    def import_csv(
        file_path_or_buffer,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Import CSV file with PV measurement data.

        Args:
            file_path_or_buffer: File path or file-like buffer
            column_mapping: Optional mapping of file columns to standard names
                          e.g., {'G': 'irradiance', 'T': 'temperature', 'P': 'power'}

        Returns:
            DataFrame with standardized column names
        """
        # Read CSV
        df = pd.read_csv(file_path_or_buffer)

        # Apply column mapping if provided
        if column_mapping:
            df = df.rename(columns=column_mapping)

        return df

    @staticmethod
    def export_results_to_csv(
        results: Dict,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export uncertainty analysis results to CSV format.

        Args:
            results: Results dictionary from uncertainty analysis
            output_path: Optional file path to save CSV

        Returns:
            CSV string
        """
        # Prepare data for export
        export_data = {}

        # Basic results
        if 'power' in results:
            export_data['Measured Power (W)'] = [results['power']]
        if 'performance_ratio' in results:
            export_data['Performance Ratio'] = [results['performance_ratio']]

        if 'combined_uncertainty' in results:
            export_data['Combined Uncertainty'] = [results['combined_uncertainty']]
        if 'expanded_uncertainty_k2' in results:
            export_data['Expanded Uncertainty (k=2)'] = [results['expanded_uncertainty_k2']]
        if 'relative_uncertainty_percent' in results:
            export_data['Relative Uncertainty (%)'] = [results['relative_uncertainty_percent']]

        # Confidence intervals
        if 'confidence_interval_95' in results:
            ci = results['confidence_interval_95']
            export_data['95% CI Lower'] = [ci[0]]
            export_data['95% CI Upper'] = [ci[1]]

        # Create DataFrame
        summary_df = pd.DataFrame(export_data)

        # Add uncertainty budget if available
        if 'budget' in results and 'components' in results['budget']:
            components = results['budget']['components']
            budget_df = pd.DataFrame(components)

            # Combine DataFrames
            summary_csv = summary_df.to_csv(index=False)
            budget_csv = "\n\nUncertainty Budget:\n" + budget_df.to_csv(index=False)
            csv_string = summary_csv + budget_csv
        else:
            csv_string = summary_df.to_csv(index=False)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(csv_string)

        return csv_string

    @staticmethod
    def create_sample_data() -> pd.DataFrame:
        """
        Create sample PV measurement data for testing.

        Returns:
            DataFrame with sample data
        """
        np.random.seed(42)

        n_samples = 100

        # Generate realistic PV data
        # Irradiance follows a pattern with some variation
        base_irradiance = 800 + 200 * np.sin(np.linspace(0, 2*np.pi, n_samples))
        irradiance = base_irradiance + np.random.normal(0, 50, n_samples)
        irradiance = np.maximum(irradiance, 0)  # No negative irradiance

        # Temperature varies with irradiance
        temperature = 25 + (irradiance / 1000) * 20 + np.random.normal(0, 2, n_samples)

        # Power calculation with efficiency and temperature effects
        efficiency = 0.20
        temp_coefficient = -0.004
        temp_effect = 1 + temp_coefficient * (temperature - 25)
        ideal_power = irradiance * efficiency * temp_effect

        # Add measurement noise
        power = ideal_power + np.random.normal(0, 5, n_samples)
        power = np.maximum(power, 0)  # No negative power

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1h'),
            'irradiance': irradiance,
            'temperature': temperature,
            'power': power
        })

        return df

    @staticmethod
    def calculate_statistics(df: pd.DataFrame, column: str) -> Dict:
        """
        Calculate basic statistics for a column.

        Args:
            df: DataFrame
            column: Column name

        Returns:
            Dictionary with statistics
        """
        if column not in df.columns:
            return {}

        data = df[column].dropna()

        if len(data) == 0:
            return {}

        stats = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75)
        }

        return stats

    @staticmethod
    def prepare_time_series_data(
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare time series data for analysis.

        Args:
            df: DataFrame with time series data
            timestamp_col: Name of timestamp column
            value_cols: List of value columns to process

        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()

        # Convert timestamp to datetime if needed
        if timestamp_col in df_processed.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_processed[timestamp_col]):
                df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col])

            # Set as index
            df_processed = df_processed.set_index(timestamp_col)

        # Sort by index
        df_processed = df_processed.sort_index()

        # Interpolate missing values if specified
        if value_cols:
            for col in value_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].interpolate(method='time', limit=5)

        return df_processed


class ResultsFormatter:
    """
    Format uncertainty analysis results for display.
    """

    @staticmethod
    def format_uncertainty_result(result: Dict, precision: int = 4) -> str:
        """
        Format uncertainty analysis result as text.

        Args:
            result: Results dictionary
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        lines = []

        lines.append("=== Uncertainty Analysis Results ===\n")

        # Main result
        if 'power' in result:
            lines.append(f"Measured Power: {result['power']:.{precision}f} W")
        if 'performance_ratio' in result:
            lines.append(f"Performance Ratio: {result['performance_ratio']:.{precision}f}")

        # Uncertainties
        if 'combined_uncertainty' in result:
            lines.append(f"\nCombined Standard Uncertainty: {result['combined_uncertainty']:.{precision}f}")
        if 'expanded_uncertainty_k2' in result:
            lines.append(f"Expanded Uncertainty (k=2): {result['expanded_uncertainty_k2']:.{precision}f}")
        if 'relative_uncertainty_percent' in result:
            lines.append(f"Relative Uncertainty: {result['relative_uncertainty_percent']:.{precision}f}%")

        # Confidence interval
        if 'confidence_interval_95' in result:
            ci = result['confidence_interval_95']
            lines.append(f"\n95% Confidence Interval: [{ci[0]:.{precision}f}, {ci[1]:.{precision}f}]")

        # Uncertainty budget
        if 'budget' in result and 'components' in result['budget']:
            lines.append("\n=== Uncertainty Budget ===")
            for comp in result['budget']['components']:
                lines.append(
                    f"\n{comp['name']}:"
                    f"\n  Standard Uncertainty: {comp['standard_uncertainty']:.{precision}f}"
                    f"\n  Contribution: {comp['percentage_contribution']:.2f}%"
                )

        return '\n'.join(lines)

    @staticmethod
    def create_summary_table(result: Dict) -> pd.DataFrame:
        """
        Create a summary table from results.

        Args:
            result: Results dictionary

        Returns:
            DataFrame with summary
        """
        data = []

        if 'power' in result:
            data.append(['Measured Power (W)', f"{result['power']:.4f}"])
        if 'performance_ratio' in result:
            data.append(['Performance Ratio', f"{result['performance_ratio']:.4f}"])
        if 'combined_uncertainty' in result:
            data.append(['Combined Uncertainty', f"{result['combined_uncertainty']:.4f}"])
        if 'expanded_uncertainty_k2' in result:
            data.append(['Expanded Uncertainty (k=2)', f"{result['expanded_uncertainty_k2']:.4f}"])
        if 'relative_uncertainty_percent' in result:
            data.append(['Relative Uncertainty (%)', f"{result['relative_uncertainty_percent']:.2f}"])

        df = pd.DataFrame(data, columns=['Metric', 'Value'])
        return df
