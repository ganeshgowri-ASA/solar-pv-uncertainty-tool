"""
File Upload and Data Extraction Utilities
Handles various file formats for PV test data, calibration certificates, and configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
import io
from PIL import Image
import PyPDF2


class DataValidator:
    """Validate extracted PV measurement data."""

    @staticmethod
    def validate_iv_ratios(vmp: float, voc: float, imp: float, isc: float) -> Dict[str, Any]:
        """
        Validate I-V curve ratios for authenticity.

        Args:
            vmp: Voltage at maximum power (V)
            voc: Open circuit voltage (V)
            imp: Current at maximum power (A)
            isc: Short circuit current (A)

        Returns:
            Dictionary with validation results
        """
        results = {"valid": True, "warnings": [], "ratios": {}}

        # Calculate ratios
        vmp_voc_ratio = vmp / voc if voc > 0 else 0
        imp_isc_ratio = imp / isc if isc > 0 else 0

        results["ratios"]["vmp_voc"] = vmp_voc_ratio
        results["ratios"]["imp_isc"] = imp_isc_ratio

        # Typical ranges
        # Vmp/Voc typically 0.75-0.85 for crystalline silicon
        # Imp/Isc typically 0.90-0.98
        if not (0.70 <= vmp_voc_ratio <= 0.90):
            results["warnings"].append(
                f"Vmp/Voc ratio ({vmp_voc_ratio:.3f}) outside typical range [0.70-0.90]"
            )
            results["valid"] = False

        if not (0.85 <= imp_isc_ratio <= 0.99):
            results["warnings"].append(
                f"Imp/Isc ratio ({imp_isc_ratio:.3f}) outside typical range [0.85-0.99]"
            )
            results["valid"] = False

        return results

    @staticmethod
    def calculate_fill_factor(vmp: float, voc: float, imp: float, isc: float) -> float:
        """Calculate fill factor."""
        if voc > 0 and isc > 0:
            return (vmp * imp) / (voc * isc)
        return 0.0

    @staticmethod
    def validate_temperature_coefficients(
        alpha_isc: float, beta_voc: float, gamma_pmax: float
    ) -> Dict[str, Any]:
        """
        Validate temperature coefficients.

        Args:
            alpha_isc: Isc temperature coefficient (%/°C)
            beta_voc: Voc temperature coefficient (%/°C)
            gamma_pmax: Pmax temperature coefficient (%/°C)

        Returns:
            Dictionary with validation results
        """
        results = {"valid": True, "warnings": []}

        # Typical ranges for crystalline silicon
        # α_Isc: +0.03 to +0.06 %/°C
        # β_Voc: -0.25 to -0.40 %/°C
        # γ_Pmax: -0.30 to -0.50 %/°C

        if not (0.02 <= alpha_isc <= 0.10):
            results["warnings"].append(
                f"α_Isc ({alpha_isc:.3f}%/°C) outside typical range [0.02-0.10]"
            )

        if not (-0.50 <= beta_voc <= -0.20):
            results["warnings"].append(
                f"β_Voc ({beta_voc:.3f}%/°C) outside typical range [-0.50 to -0.20]"
            )

        if not (-0.60 <= gamma_pmax <= -0.25):
            results["warnings"].append(
                f"γ_Pmax ({gamma_pmax:.3f}%/°C) outside typical range [-0.60 to -0.25]"
            )

        results["valid"] = len(results["warnings"]) == 0

        return results


class PDFExtractor:
    """Extract data from PDF calibration certificates and test reports."""

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extract all text from PDF file.

        Args:
            pdf_file: Uploaded PDF file object

        Returns:
            Extracted text string
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"

    @staticmethod
    def extract_calibration_data(text: str) -> Dict[str, Any]:
        """
        Extract calibration data from certificate text.

        Args:
            text: Extracted PDF text

        Returns:
            Dictionary with calibration parameters
        """
        data = {
            "isc_ref": None,
            "voc_ref": None,
            "pmax_ref": None,
            "uncertainty": None,
            "calibration_date": None,
            "lab": None
        }

        # Regex patterns for common formats
        patterns = {
            "isc_ref": r"I[sS][cC].*?(\d+\.?\d*)\s*A",
            "voc_ref": r"V[oO][cC].*?(\d+\.?\d*)\s*V",
            "pmax_ref": r"P[mM][aA][xX].*?(\d+\.?\d*)\s*W",
            "uncertainty": r"[uU]ncertainty.*?(\d+\.?\d*)\s*%",
            "calibration_date": r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    data[key] = float(match.group(1))
                except (ValueError, IndexError):
                    data[key] = match.group(1) if match.groups() else None

        return data


class ExcelExtractor:
    """Extract data from Excel files (I-V curves, summary data)."""

    @staticmethod
    def read_iv_curve_data(excel_file) -> Optional[pd.DataFrame]:
        """
        Read I-V curve data from Excel file.

        Args:
            excel_file: Uploaded Excel file object

        Returns:
            DataFrame with voltage and current columns
        """
        try:
            df = pd.read_excel(excel_file)

            # Try to identify voltage and current columns
            voltage_col = None
            current_col = None

            for col in df.columns:
                col_lower = str(col).lower()
                if any(v in col_lower for v in ['voltage', 'v', 'vmp', 'voc']):
                    voltage_col = col
                if any(i in col_lower for i in ['current', 'i', 'imp', 'isc', 'a']):
                    current_col = col

            if voltage_col and current_col:
                return df[[voltage_col, current_col]].rename(
                    columns={voltage_col: 'Voltage', current_col: 'Current'}
                )
            else:
                # Return first two numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    return df[numeric_cols[:2]].rename(
                        columns={numeric_cols[0]: 'Voltage', numeric_cols[1]: 'Current'}
                    )

        except Exception as e:
            print(f"Error reading Excel: {e}")

        return None

    @staticmethod
    def extract_summary_data(excel_file) -> Dict[str, Any]:
        """
        Extract summary measurement data from Excel.

        Args:
            excel_file: Uploaded Excel file object

        Returns:
            Dictionary with extracted parameters
        """
        data = {}

        try:
            df = pd.read_excel(excel_file)

            # Look for key parameters in the DataFrame
            search_terms = {
                "Pmax": ["pmax", "p_max", "power", "mpp"],
                "Voc": ["voc", "v_oc", "open circuit voltage"],
                "Isc": ["isc", "i_sc", "short circuit current"],
                "Vmp": ["vmp", "v_mp", "vmpp"],
                "Imp": ["imp", "i_mp", "impp"],
                "FF": ["ff", "fill factor", "fillfactor"],
                "Efficiency": ["eff", "efficiency", "eta"]
            }

            for param, terms in search_terms.items():
                for term in terms:
                    # Search in column names
                    for col in df.columns:
                        if term in str(col).lower():
                            # Get first numeric value
                            numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                            if not numeric_values.empty:
                                data[param] = float(numeric_values.iloc[0])
                                break
                    if param in data:
                        break

        except Exception as e:
            print(f"Error extracting summary data: {e}")

        return data


class DatasheetExtractor:
    """Extract module specifications from datasheet text or structured data."""

    @staticmethod
    def parse_module_datasheet(text: str) -> Dict[str, Any]:
        """
        Parse module datasheet to extract key specifications.

        Args:
            text: Datasheet text content

        Returns:
            Dictionary with module specifications
        """
        specs = {
            "pmax_stc": None,
            "voc_stc": None,
            "isc_stc": None,
            "vmp_stc": None,
            "imp_stc": None,
            "efficiency": None,
            "temp_coeff_pmax": None,
            "temp_coeff_voc": None,
            "temp_coeff_isc": None,
            "cell_count": None,
            "technology": None,
            "dimensions": None,
            "weight": None
        }

        # Regex patterns for datasheet extraction
        patterns = {
            "pmax_stc": r"P[mM][aA][xX].*?(\d+\.?\d*)\s*W",
            "voc_stc": r"V[oO][cC].*?(\d+\.?\d*)\s*V",
            "isc_stc": r"I[sS][cC].*?(\d+\.?\d*)\s*A",
            "vmp_stc": r"V[mM][pP].*?(\d+\.?\d*)\s*V",
            "imp_stc": r"I[mM][pP].*?(\d+\.?\d*)\s*A",
            "efficiency": r"[eE]fficiency.*?(\d+\.?\d*)\s*%",
            "temp_coeff_pmax": r"[tT]emp.*?[cC]oeff.*?P.*?(-?\d+\.?\d*)\s*%",
            "temp_coeff_voc": r"[tT]emp.*?[cC]oeff.*?V.*?(-?\d+\.?\d*)\s*%",
            "temp_coeff_isc": r"[tT]emp.*?[cC]oeff.*?I.*?(\+?\d+\.?\d*)\s*%",
            "cell_count": r"(\d+)\s*[cC]ells?"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    specs[key] = float(match.group(1))
                except (ValueError, IndexError):
                    pass

        # Detect technology
        tech_keywords = {
            "PERC": ["perc", "passivated"],
            "TOPCon": ["topcon", "tunnel oxide"],
            "HJT": ["hjt", "heterojunction", "hit"],
            "Perovskite": ["perovskite"],
            "CIGS": ["cigs", "copper indium"],
            "CdTe": ["cdte", "cadmium telluride"]
        }

        for tech, keywords in tech_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                specs["technology"] = tech
                break

        return specs


class PVsystPANParser:
    """Parse PVsyst .PAN files for module parameters."""

    @staticmethod
    def parse_pan_file(pan_file) -> Dict[str, Any]:
        """
        Parse PVsyst .PAN file.

        Args:
            pan_file: Uploaded .PAN file object

        Returns:
            Dictionary with module parameters
        """
        data = {
            "module_name": None,
            "technology": None,
            "pmax_stc": None,
            "voc_stc": None,
            "isc_stc": None,
            "vmp_stc": None,
            "imp_stc": None,
            "cells_series": None,
            "cells_parallel": None,
            "temp_coeff_pmax": None,
            "temp_coeff_voc": None,
            "temp_coeff_isc": None,
            "noct": None,
            "bifaciality": None
        }

        try:
            # Read file content
            content = pan_file.read().decode('utf-8', errors='ignore')

            # Parse key-value pairs
            patterns = {
                "module_name": r"PVObject.*?pvModule.*?\"(.*?)\"",
                "pmax_stc": r"Pmpp\s*=\s*(\d+\.?\d*)",
                "voc_stc": r"Voc\s*=\s*(\d+\.?\d*)",
                "isc_stc": r"Isc\s*=\s*(\d+\.?\d*)",
                "vmp_stc": r"Vmpp\s*=\s*(\d+\.?\d*)",
                "imp_stc": r"Impp\s*=\s*(\d+\.?\d*)",
                "cells_series": r"NCelS\s*=\s*(\d+)",
                "cells_parallel": r"NCelP\s*=\s*(\d+)",
                "temp_coeff_pmax": r"muPmpp\s*=\s*(-?\d+\.?\d*)",
                "temp_coeff_voc": r"muVocSpec\s*=\s*(-?\d+\.?\d*)",
                "temp_coeff_isc": r"muIscSpec\s*=\s*(\+?\d+\.?\d*)",
                "noct": r"TRef\s*=\s*(\d+\.?\d*)",
                "bifaciality": r"BifacialityFactor\s*=\s*(\d+\.?\d*)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        data[key] = float(match.group(1)) if key != "module_name" else match.group(1)
                    except (ValueError, IndexError):
                        data[key] = match.group(1)

        except Exception as e:
            print(f"Error parsing PAN file: {e}")

        return data


def create_repeatability_dataframe(measurements: List[Dict[str, float]]) -> pd.DataFrame:
    """
    Create DataFrame from repeatability measurements.

    Args:
        measurements: List of dictionaries with measurement data

    Returns:
        DataFrame with statistics
    """
    df = pd.DataFrame(measurements)

    # Calculate statistics
    stats = {
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
        "range": df.max() - df.min(),
        "cv_percent": (df.std() / df.mean()) * 100  # Coefficient of variation
    }

    return pd.DataFrame(stats)
