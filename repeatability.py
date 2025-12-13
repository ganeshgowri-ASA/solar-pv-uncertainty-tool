"""
Repeatability Analysis Module for PV Measurement Uncertainty Tool
Handles multi-file upload, statistical calculations, and Type A uncertainty computation.
Implements repeatability analysis per GUM (JCGM 100:2008) methodology.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import io
import re


@dataclass
class IVMeasurement:
    """Single IV measurement data from a file."""
    filename: str
    measurement_id: int
    timestamp: Optional[datetime] = None
    isc: Optional[float] = None  # Short-circuit current (A)
    voc: Optional[float] = None  # Open-circuit voltage (V)
    pmax: Optional[float] = None  # Maximum power (W)
    vmp: Optional[float] = None  # Voltage at Pmax (V)
    imp: Optional[float] = None  # Current at Pmax (A)
    fill_factor: Optional[float] = None  # Fill factor
    irradiance: Optional[float] = None  # Measured irradiance (W/m²)
    temperature: Optional[float] = None  # Module temperature (°C)
    raw_data: Optional[pd.DataFrame] = None  # Raw IV curve data


@dataclass
class RepeatabilityStatistics:
    """Statistical results from repeatability analysis."""
    parameter: str
    n_measurements: int
    mean: float
    std_dev: float
    variance: float
    cv_percent: float  # Coefficient of variation (%)
    min_value: float
    max_value: float
    range_value: float
    standard_uncertainty: float  # Type A uncertainty: std_dev / sqrt(n)
    expanded_uncertainty_k2: float
    relative_uncertainty_percent: float
    values: List[float] = field(default_factory=list)


@dataclass
class RepeatabilityResult:
    """Complete repeatability analysis result."""
    analysis_id: str
    analysis_timestamp: datetime
    n_files: int
    valid_measurements: int
    module_id: Optional[str] = None
    statistics: Dict[str, RepeatabilityStatistics] = field(default_factory=dict)
    outliers_detected: List[Dict] = field(default_factory=list)
    type_a_uncertainty: Optional[Dict[str, float]] = None
    validation_passed: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RepeatabilityFileParser:
    """Parse IV curve data from various file formats."""

    SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']

    # Common column name patterns for IV parameters
    COLUMN_PATTERNS = {
        'isc': [r'i[_\-]?sc', r'short[_\-]?circuit', r'isc\s*\(a\)', r'i_sc'],
        'voc': [r'v[_\-]?oc', r'open[_\-]?circuit', r'voc\s*\(v\)', r'v_oc'],
        'pmax': [r'p[_\-]?max', r'p[_\-]?mpp', r'max[_\-]?power', r'pmax\s*\(w\)', r'power'],
        'vmp': [r'v[_\-]?mp', r'v[_\-]?mpp', r'vmp\s*\(v\)'],
        'imp': [r'i[_\-]?mp', r'i[_\-]?mpp', r'imp\s*\(a\)'],
        'ff': [r'fill[_\-]?factor', r'ff', r'ff\s*\(%\)'],
        'irradiance': [r'irrad', r'g\s*\(w', r'irradiance', r'intensity'],
        'temperature': [r'temp', r't\s*\(', r'temperature', r'module[_\-]?temp'],
        'voltage': [r'^v$', r'voltage', r'v\s*\(v\)'],
        'current': [r'^i$', r'current', r'i\s*\(a\)'],
    }

    @classmethod
    def parse_file(cls, file_obj, filename: str) -> Optional[IVMeasurement]:
        """
        Parse a single file and extract IV measurement data.

        Args:
            file_obj: File object (from Streamlit uploader or file path)
            filename: Name of the file

        Returns:
            IVMeasurement object or None if parsing fails
        """
        try:
            # Determine file type
            ext = '.' + filename.split('.')[-1].lower()
            if ext not in cls.SUPPORTED_EXTENSIONS:
                return None

            # Read file based on extension
            if ext == '.csv':
                df = pd.read_csv(file_obj)
            else:  # Excel formats
                df = pd.read_excel(file_obj)

            # Clean column names
            df.columns = df.columns.str.strip().str.lower()

            # Extract parameters
            measurement = IVMeasurement(
                filename=filename,
                measurement_id=0,
                timestamp=datetime.now()
            )

            # Map columns to parameters
            column_mapping = cls._find_column_mapping(df.columns.tolist())

            # Extract summary parameters if available
            for param, col in column_mapping.items():
                if col and col in df.columns:
                    # Get first non-null value or mean for summary parameters
                    values = pd.to_numeric(df[col], errors='coerce').dropna()
                    if not values.empty:
                        if param in ['isc', 'voc', 'pmax', 'vmp', 'imp', 'ff', 'irradiance', 'temperature']:
                            # For summary params, take first value or mean
                            value = values.iloc[0] if len(values) == 1 else values.mean()
                            setattr(measurement, param if param != 'ff' else 'fill_factor', value)

            # Store raw data for IV curve
            if 'voltage' in column_mapping and 'current' in column_mapping:
                v_col = column_mapping['voltage']
                i_col = column_mapping['current']
                if v_col and i_col:
                    measurement.raw_data = df[[v_col, i_col]].rename(
                        columns={v_col: 'voltage', i_col: 'current'}
                    )

            # Calculate derived parameters if missing
            measurement = cls._calculate_derived_parameters(measurement)

            return measurement

        except Exception as e:
            print(f"Error parsing file {filename}: {str(e)}")
            return None

    @classmethod
    def _find_column_mapping(cls, columns: List[str]) -> Dict[str, Optional[str]]:
        """Find column names matching expected parameters."""
        mapping = {}

        for param, patterns in cls.COLUMN_PATTERNS.items():
            mapping[param] = None
            for pattern in patterns:
                for col in columns:
                    if re.search(pattern, col, re.IGNORECASE):
                        mapping[param] = col
                        break
                if mapping[param]:
                    break

        return mapping

    @classmethod
    def _calculate_derived_parameters(cls, measurement: IVMeasurement) -> IVMeasurement:
        """Calculate derived parameters from raw IV curve if available."""

        # Calculate Pmax from Vmp and Imp if not present
        if measurement.pmax is None and measurement.vmp and measurement.imp:
            measurement.pmax = measurement.vmp * measurement.imp

        # Calculate Fill Factor if not present
        if measurement.fill_factor is None:
            if all([measurement.voc, measurement.isc, measurement.vmp, measurement.imp]):
                if measurement.voc > 0 and measurement.isc > 0:
                    measurement.fill_factor = (measurement.vmp * measurement.imp) / (measurement.voc * measurement.isc)

        # Extract parameters from IV curve if available and summary params missing
        if measurement.raw_data is not None and len(measurement.raw_data) > 5:
            df = measurement.raw_data
            v = df['voltage'].values
            i = df['current'].values

            # Calculate Isc (current at V=0)
            if measurement.isc is None:
                isc_idx = np.abs(v).argmin()
                measurement.isc = abs(i[isc_idx])

            # Calculate Voc (voltage at I=0)
            if measurement.voc is None:
                voc_idx = np.abs(i).argmin()
                measurement.voc = abs(v[voc_idx])

            # Calculate Pmax, Vmp, Imp
            if measurement.pmax is None:
                power = np.abs(v * i)
                max_idx = power.argmax()
                measurement.pmax = power[max_idx]
                measurement.vmp = abs(v[max_idx])
                measurement.imp = abs(i[max_idx])

        return measurement


class RepeatabilityAnalyzer:
    """
    Analyze repeatability from multiple measurements.
    Calculates Type A uncertainty per GUM methodology.
    """

    MINIMUM_FILES_REQUIRED = 10
    PARAMETERS = ['isc', 'voc', 'pmax', 'vmp', 'imp', 'fill_factor']

    def __init__(self, minimum_files: int = 10):
        """
        Initialize analyzer.

        Args:
            minimum_files: Minimum number of files required for valid analysis
        """
        self.minimum_files = minimum_files
        self.measurements: List[IVMeasurement] = []
        self.result: Optional[RepeatabilityResult] = None

    def add_measurement(self, measurement: IVMeasurement) -> None:
        """Add a measurement to the analysis."""
        measurement.measurement_id = len(self.measurements) + 1
        self.measurements.append(measurement)

    def add_files(self, file_objects: List[Tuple[Any, str]]) -> int:
        """
        Parse and add multiple files.

        Args:
            file_objects: List of (file_object, filename) tuples

        Returns:
            Number of successfully parsed files
        """
        success_count = 0
        for file_obj, filename in file_objects:
            measurement = RepeatabilityFileParser.parse_file(file_obj, filename)
            if measurement:
                self.add_measurement(measurement)
                success_count += 1
        return success_count

    def analyze(self, module_id: Optional[str] = None) -> RepeatabilityResult:
        """
        Perform repeatability analysis on all measurements.

        Args:
            module_id: Optional module identifier

        Returns:
            RepeatabilityResult with statistics and Type A uncertainty
        """
        result = RepeatabilityResult(
            analysis_id=f"REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            analysis_timestamp=datetime.now(),
            n_files=len(self.measurements),
            valid_measurements=0,
            module_id=module_id
        )

        # Validate minimum files
        if len(self.measurements) < self.minimum_files:
            result.errors.append(
                f"Insufficient measurements: {len(self.measurements)} files provided, "
                f"minimum {self.minimum_files} required"
            )
            result.validation_passed = False
            self.result = result
            return result

        # Calculate statistics for each parameter
        type_a_uncertainties = {}

        for param in self.PARAMETERS:
            values = []
            for m in self.measurements:
                val = getattr(m, param, None)
                if val is not None and not np.isnan(val):
                    values.append(val)

            if len(values) >= 2:
                stats = self._calculate_statistics(param, values)
                result.statistics[param] = stats

                # Store Type A uncertainty for Pmax (primary parameter)
                type_a_uncertainties[param] = stats.relative_uncertainty_percent

        # Set primary Type A uncertainty (from Pmax)
        result.type_a_uncertainty = type_a_uncertainties

        # Count valid measurements (those with Pmax)
        result.valid_measurements = sum(
            1 for m in self.measurements if m.pmax is not None
        )

        # Detect outliers using IQR method
        result.outliers_detected = self._detect_outliers()

        # Validate results
        result.validation_passed = (
            result.valid_measurements >= self.minimum_files and
            len(result.errors) == 0
        )

        # Add warnings for high variability
        if 'pmax' in result.statistics:
            pmax_stats = result.statistics['pmax']
            if pmax_stats.cv_percent > 2.0:
                result.warnings.append(
                    f"High variability detected: CV = {pmax_stats.cv_percent:.2f}% "
                    f"(typical: <1% for good repeatability)"
                )

        self.result = result
        return result

    def _calculate_statistics(self, parameter: str, values: List[float]) -> RepeatabilityStatistics:
        """Calculate comprehensive statistics for a parameter."""
        n = len(values)
        values_array = np.array(values)

        mean = np.mean(values_array)
        std_dev = np.std(values_array, ddof=1)  # Sample standard deviation
        variance = np.var(values_array, ddof=1)

        # Type A uncertainty: standard uncertainty of the mean
        # u_A = s / sqrt(n) where s is sample standard deviation
        standard_uncertainty = std_dev / np.sqrt(n)
        expanded_uncertainty_k2 = standard_uncertainty * 2.0

        # Relative uncertainty as percentage of mean
        relative_uncertainty = (standard_uncertainty / mean * 100) if mean != 0 else 0

        # Coefficient of variation
        cv_percent = (std_dev / mean * 100) if mean != 0 else 0

        return RepeatabilityStatistics(
            parameter=parameter,
            n_measurements=n,
            mean=mean,
            std_dev=std_dev,
            variance=variance,
            cv_percent=cv_percent,
            min_value=np.min(values_array),
            max_value=np.max(values_array),
            range_value=np.max(values_array) - np.min(values_array),
            standard_uncertainty=standard_uncertainty,
            expanded_uncertainty_k2=expanded_uncertainty_k2,
            relative_uncertainty_percent=relative_uncertainty,
            values=values
        )

    def _detect_outliers(self, iqr_multiplier: float = 1.5) -> List[Dict]:
        """
        Detect outliers using IQR method.

        Args:
            iqr_multiplier: Multiplier for IQR (default 1.5 for standard outliers)

        Returns:
            List of detected outliers with details
        """
        outliers = []

        if 'pmax' not in self.result.statistics:
            return outliers

        values = np.array(self.result.statistics['pmax'].values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        for i, (m, val) in enumerate(zip(self.measurements, values)):
            if val < lower_bound or val > upper_bound:
                outliers.append({
                    'measurement_id': m.measurement_id,
                    'filename': m.filename,
                    'parameter': 'pmax',
                    'value': val,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'reason': 'outside_iqr'
                })

        return outliers

    def get_type_a_uncertainty_for_budget(self) -> float:
        """
        Get Type A uncertainty (as %) for integration with uncertainty budget.
        Returns the repeatability contribution from Pmax measurements.
        """
        if self.result and 'pmax' in self.result.statistics:
            return self.result.statistics['pmax'].relative_uncertainty_percent
        return 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Convert measurements to pandas DataFrame."""
        data = []
        for m in self.measurements:
            data.append({
                'File': m.filename,
                'Measurement ID': m.measurement_id,
                'Isc (A)': m.isc,
                'Voc (V)': m.voc,
                'Pmax (W)': m.pmax,
                'Vmp (V)': m.vmp,
                'Imp (A)': m.imp,
                'Fill Factor': m.fill_factor,
                'Irradiance (W/m²)': m.irradiance,
                'Temperature (°C)': m.temperature
            })
        return pd.DataFrame(data)

    def get_statistics_dataframe(self) -> pd.DataFrame:
        """Get statistics as pandas DataFrame."""
        if not self.result:
            return pd.DataFrame()

        data = []
        for param, stats in self.result.statistics.items():
            data.append({
                'Parameter': param.upper().replace('_', ' '),
                'N': stats.n_measurements,
                'Mean': f"{stats.mean:.4f}",
                'Std Dev': f"{stats.std_dev:.4f}",
                'CV (%)': f"{stats.cv_percent:.3f}",
                'Min': f"{stats.min_value:.4f}",
                'Max': f"{stats.max_value:.4f}",
                'Range': f"{stats.range_value:.4f}",
                'Type A Unc (%)': f"{stats.relative_uncertainty_percent:.4f}",
                'Expanded Unc (k=2)': f"{stats.expanded_uncertainty_k2:.4f}"
            })
        return pd.DataFrame(data)


class RepeatabilityDatabaseHandler:
    """Handle database operations for repeatability analysis."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database handler.

        Args:
            database_url: PostgreSQL connection URL (from Railway)
        """
        self.database_url = database_url
        self.connection = None
        self._engine = None

    def connect(self) -> bool:
        """Establish database connection."""
        if not self.database_url:
            return False

        try:
            from sqlalchemy import create_engine
            self._engine = create_engine(self.database_url)
            self.connection = self._engine.connect()
            return True
        except ImportError:
            print("SQLAlchemy not installed. Database features disabled.")
            return False
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def create_tables(self) -> bool:
        """Create required tables if they don't exist."""
        if not self._engine:
            return False

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS measurement_files (
            id SERIAL PRIMARY KEY,
            analysis_id VARCHAR(100) NOT NULL,
            filename VARCHAR(255) NOT NULL,
            measurement_id INTEGER,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            isc FLOAT,
            voc FLOAT,
            pmax FLOAT,
            vmp FLOAT,
            imp FLOAT,
            fill_factor FLOAT,
            irradiance FLOAT,
            temperature FLOAT,
            file_content BYTEA
        );

        CREATE TABLE IF NOT EXISTS repeatability_results (
            id SERIAL PRIMARY KEY,
            analysis_id VARCHAR(100) UNIQUE NOT NULL,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            module_id VARCHAR(100),
            n_files INTEGER,
            valid_measurements INTEGER,
            pmax_mean FLOAT,
            pmax_std_dev FLOAT,
            pmax_cv_percent FLOAT,
            type_a_uncertainty_percent FLOAT,
            validation_passed BOOLEAN,
            warnings TEXT,
            errors TEXT,
            linked_uncertainty_result_id INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_measurement_files_analysis_id
        ON measurement_files(analysis_id);

        CREATE INDEX IF NOT EXISTS idx_repeatability_results_module_id
        ON repeatability_results(module_id);
        """

        try:
            with self._engine.begin() as conn:
                conn.execute(create_table_sql)
            return True
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False

    def save_measurement(self, measurement: IVMeasurement, analysis_id: str,
                        file_content: Optional[bytes] = None) -> Optional[int]:
        """Save a measurement to the database."""
        if not self._engine:
            return None

        insert_sql = """
        INSERT INTO measurement_files
        (analysis_id, filename, measurement_id, isc, voc, pmax, vmp, imp,
         fill_factor, irradiance, temperature, file_content)
        VALUES (%(analysis_id)s, %(filename)s, %(measurement_id)s, %(isc)s,
                %(voc)s, %(pmax)s, %(vmp)s, %(imp)s, %(fill_factor)s,
                %(irradiance)s, %(temperature)s, %(file_content)s)
        RETURNING id;
        """

        try:
            with self._engine.begin() as conn:
                result = conn.execute(insert_sql, {
                    'analysis_id': analysis_id,
                    'filename': measurement.filename,
                    'measurement_id': measurement.measurement_id,
                    'isc': measurement.isc,
                    'voc': measurement.voc,
                    'pmax': measurement.pmax,
                    'vmp': measurement.vmp,
                    'imp': measurement.imp,
                    'fill_factor': measurement.fill_factor,
                    'irradiance': measurement.irradiance,
                    'temperature': measurement.temperature,
                    'file_content': file_content
                })
                return result.fetchone()[0]
        except Exception as e:
            print(f"Error saving measurement: {e}")
            return None

    def save_repeatability_result(self, result: RepeatabilityResult,
                                  uncertainty_result_id: Optional[int] = None) -> Optional[int]:
        """Save repeatability analysis result to database."""
        if not self._engine:
            return None

        pmax_stats = result.statistics.get('pmax')

        insert_sql = """
        INSERT INTO repeatability_results
        (analysis_id, analysis_timestamp, module_id, n_files, valid_measurements,
         pmax_mean, pmax_std_dev, pmax_cv_percent, type_a_uncertainty_percent,
         validation_passed, warnings, errors, linked_uncertainty_result_id)
        VALUES (%(analysis_id)s, %(analysis_timestamp)s, %(module_id)s,
                %(n_files)s, %(valid_measurements)s, %(pmax_mean)s,
                %(pmax_std_dev)s, %(pmax_cv_percent)s, %(type_a_uncertainty)s,
                %(validation_passed)s, %(warnings)s, %(errors)s, %(linked_id)s)
        RETURNING id;
        """

        try:
            with self._engine.begin() as conn:
                db_result = conn.execute(insert_sql, {
                    'analysis_id': result.analysis_id,
                    'analysis_timestamp': result.analysis_timestamp,
                    'module_id': result.module_id,
                    'n_files': result.n_files,
                    'valid_measurements': result.valid_measurements,
                    'pmax_mean': pmax_stats.mean if pmax_stats else None,
                    'pmax_std_dev': pmax_stats.std_dev if pmax_stats else None,
                    'pmax_cv_percent': pmax_stats.cv_percent if pmax_stats else None,
                    'type_a_uncertainty': pmax_stats.relative_uncertainty_percent if pmax_stats else None,
                    'validation_passed': result.validation_passed,
                    'warnings': '\n'.join(result.warnings),
                    'errors': '\n'.join(result.errors),
                    'linked_id': uncertainty_result_id
                })
                return db_result.fetchone()[0]
        except Exception as e:
            print(f"Error saving result: {e}")
            return None

    def get_repeatability_history(self, module_id: Optional[str] = None,
                                  limit: int = 10) -> pd.DataFrame:
        """Get historical repeatability analyses."""
        if not self._engine:
            return pd.DataFrame()

        query = """
        SELECT analysis_id, analysis_timestamp, module_id, n_files,
               pmax_mean, pmax_std_dev, pmax_cv_percent, type_a_uncertainty_percent,
               validation_passed
        FROM repeatability_results
        """

        if module_id:
            query += f" WHERE module_id = '{module_id}'"

        query += f" ORDER BY analysis_timestamp DESC LIMIT {limit}"

        try:
            return pd.read_sql(query, self._engine)
        except Exception as e:
            print(f"Error querying history: {e}")
            return pd.DataFrame()

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
        if self._engine:
            self._engine.dispose()


def calculate_repeatability_coefficient(std_dev: float, mean: float,
                                        confidence_level: float = 0.95) -> float:
    """
    Calculate repeatability coefficient (r) per ISO 5725-2.

    r = 2.8 * s_r (for 95% confidence)

    Args:
        std_dev: Standard deviation of measurements
        mean: Mean of measurements
        confidence_level: Confidence level (default 0.95)

    Returns:
        Repeatability coefficient as percentage of mean
    """
    # Factor for 95% confidence is approximately 2.8 (derived from t-distribution)
    if confidence_level == 0.95:
        factor = 2.8
    elif confidence_level == 0.99:
        factor = 3.7
    else:
        factor = 2.8  # Default to 95%

    r = factor * std_dev
    return (r / mean * 100) if mean != 0 else 0


def validate_file_format(file_obj, filename: str) -> Tuple[bool, str]:
    """
    Validate that a file is in an acceptable format.

    Args:
        file_obj: File object
        filename: Name of file

    Returns:
        Tuple of (is_valid, error_message)
    """
    ext = '.' + filename.split('.')[-1].lower()

    if ext not in RepeatabilityFileParser.SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file format: {ext}. Supported: {', '.join(RepeatabilityFileParser.SUPPORTED_EXTENSIONS)}"

    try:
        # Try to read a small portion to validate
        if ext == '.csv':
            df = pd.read_csv(file_obj, nrows=5)
        else:
            df = pd.read_excel(file_obj, nrows=5)

        if df.empty:
            return False, "File appears to be empty"

        # Reset file position for later reading
        file_obj.seek(0)
        return True, ""

    except Exception as e:
        file_obj.seek(0)  # Reset position even on error
        return False, f"Error reading file: {str(e)}"
