"""Data validation checks for quality assessment.

Validates completeness, uniqueness, referential integrity, ranges, formats,
and type consistency.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def validate_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data completeness (missing values, empty strings).
    
    Returns:
        Dictionary with completeness validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "total_columns": len(df.columns),
            "columns_with_missing": 0,
            "columns_with_empty_strings": 0,
        },
    }
    
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) if len(df) > 0 else 0
        
        # Check for empty strings
        empty_string_count = 0
        if df[col].dtype == "object":
            empty_string_count = (df[col].astype(str).str.strip() == "").sum()
        empty_string_pct = empty_string_count / len(df) if len(df) > 0 else 0
        
        if null_pct > 0 or empty_string_pct > 0:
            validation_results["summary"]["columns_with_missing"] += 1
            
            severity = "high" if null_pct > 0.5 else ("medium" if null_pct > 0.2 else "low")
            
            check = {
                "column": col,
                "type": "completeness",
                "severity": severity,
                "null_count": int(null_count),
                "null_percentage": float(null_pct),
                "empty_string_count": int(empty_string_count),
                "empty_string_percentage": float(empty_string_pct),
                "message": f"Column '{col}' has {null_pct:.1%} nulls and {empty_string_pct:.1%} empty strings",
            }
            validation_results["checks"].append(check)
    
    return validation_results


def validate_uniqueness(df: pd.DataFrame, key_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate uniqueness constraints.
    
    Args:
        df: DataFrame to validate
        key_columns: Optional list of columns that should be unique
    
    Returns:
        Dictionary with uniqueness validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "duplicate_rows": 0,
            "duplicate_percentage": 0.0,
        },
    }
    
    # Check for duplicate rows
    duplicate_count = len(df) - len(df.drop_duplicates())
    duplicate_pct = duplicate_count / len(df) if len(df) > 0 else 0
    
    validation_results["summary"]["duplicate_rows"] = duplicate_count
    validation_results["summary"]["duplicate_percentage"] = float(duplicate_pct)
    
    if duplicate_pct > 0:
        severity = "high" if duplicate_pct > 0.1 else ("medium" if duplicate_pct > 0.05 else "low")
        
        validation_results["checks"].append({
            "type": "uniqueness",
            "severity": severity,
            "duplicate_count": duplicate_count,
            "duplicate_percentage": float(duplicate_pct),
            "message": f"Found {duplicate_count} duplicate rows ({duplicate_pct:.1%} of data)",
        })
    
    # Check key columns for uniqueness
    if key_columns:
        for col in key_columns:
            if col not in df.columns:
                continue
            
            unique_count = df[col].nunique()
            total_count = len(df)
            
            if unique_count < total_count:
                duplicates = total_count - unique_count
                severity = "high" if (unique_count / total_count) < 0.9 else "medium"
                
                validation_results["checks"].append({
                    "column": col,
                    "type": "uniqueness",
                    "severity": severity,
                    "unique_count": unique_count,
                    "total_count": total_count,
                    "duplicate_count": duplicates,
                    "message": f"Column '{col}' expected to be unique but has {duplicates} duplicates",
                })
    
    return validation_results


def validate_referential_integrity(
    source_df: pd.DataFrame,
    source_column: str,
    target_df: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    """Validate referential integrity between two DataFrames.
    
    Checks for orphaned records (values in source that don't exist in target).
    
    Returns:
        Dictionary with referential integrity validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "source_column": source_column,
            "target_column": target_column,
            "orphaned_count": 0,
            "orphaned_percentage": 0.0,
        },
    }
    
    if source_column not in source_df.columns or target_column not in target_df.columns:
        return validation_results
    
    source_values = set(source_df[source_column].dropna().astype(str))
    target_values = set(target_df[target_column].dropna().astype(str))
    
    orphaned = source_values - target_values
    orphaned_count = len(orphaned)
    orphaned_pct = orphaned_count / len(source_values) if source_values else 0
    
    validation_results["summary"]["orphaned_count"] = orphaned_count
    validation_results["summary"]["orphaned_percentage"] = float(orphaned_pct)
    
    if orphaned_count > 0:
        severity = "high" if orphaned_pct > 0.1 else ("medium" if orphaned_pct > 0.05 else "low")
        
        validation_results["checks"].append({
            "type": "referential_integrity",
            "severity": severity,
            "orphaned_count": orphaned_count,
            "orphaned_percentage": float(orphaned_pct),
            "message": (
                f"Found {orphaned_count} orphaned values in '{source_column}' "
                f"that don't exist in '{target_column}' ({orphaned_pct:.1%})"
            ),
        })
    
    return validation_results


def validate_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate numeric ranges (outliers, negative values, zeros).
    
    Returns:
        Dictionary with range validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "columns_with_outliers": 0,
            "columns_with_negative_values": 0,
            "columns_with_zero_values": 0,
        },
    }
    
    # Exclude boolean dtype explicitly to avoid boolean subtraction errors
    numeric_cols = df.select_dtypes(include=[np.number], exclude=['bool']).columns
    
    for col in numeric_cols:
        series = df[col].dropna()
        
        if len(series) == 0:
            continue
        
        # Detect outliers using IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(series)
                validation_results["summary"]["columns_with_outliers"] += 1
                
                severity = "medium" if outlier_pct > 0.05 else "low"
                
                validation_results["checks"].append({
                    "column": col,
                    "type": "range",
                    "subtype": "outliers",
                    "severity": severity,
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(outlier_pct),
                    "message": f"Column '{col}' has {len(outliers)} outliers ({outlier_pct:.1%})",
                })
        
        # Check for negative values (may be unexpected)
        negative_count = (series < 0).sum()
        if negative_count > 0:
            negative_pct = negative_count / len(series)
            validation_results["summary"]["columns_with_negative_values"] += 1
            
            # Only flag if significant portion is negative
            if negative_pct > 0.01:
                validation_results["checks"].append({
                    "column": col,
                    "type": "range",
                    "subtype": "negative_values",
                    "severity": "low",
                    "negative_count": int(negative_count),
                    "negative_percentage": float(negative_pct),
                    "message": f"Column '{col}' has {negative_count} negative values ({negative_pct:.1%})",
                })
        
        # Check for zero values (may be unexpected)
        zero_count = (series == 0).sum()
        if zero_count > 0:
            zero_pct = zero_count / len(series)
            validation_results["summary"]["columns_with_zero_values"] += 1
            
            # Only flag if significant portion is zero
            if zero_pct > 0.1:
                validation_results["checks"].append({
                    "column": col,
                    "type": "range",
                    "subtype": "zero_values",
                    "severity": "low",
                    "zero_count": int(zero_count),
                    "zero_percentage": float(zero_pct),
                    "message": f"Column '{col}' has {zero_count} zero values ({zero_pct:.1%})",
                })
    
    return validation_results


def validate_formats(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data formats (dates, strings, patterns).
    
    Returns:
        Dictionary with format validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "columns_with_format_issues": 0,
        },
    }
    
    for col in df.columns:
        series = df[col]
        
        # Check date format consistency
        if series.dtype == "object":
            # Try to detect date-like strings
            date_pattern_count = 0
            total_non_null = series.notna().sum()
            
            if total_non_null > 0:
                # Sample to check format consistency
                sample_size = min(1000, total_non_null)
                sample = series.dropna().head(sample_size)
                
                # Check if values look like dates but aren't parsed as datetime
                date_like_patterns = ["-", "/", "T"]
                for val in sample:
                    val_str = str(val)
                    if any(pattern in val_str for pattern in date_like_patterns):
                        date_pattern_count += 1
                
                # If many look like dates but dtype is object, flag it
                if date_pattern_count > sample_size * 0.5 and total_non_null > 100:
                    validation_results["summary"]["columns_with_format_issues"] += 1
                    
                    validation_results["checks"].append({
                        "column": col,
                        "type": "format",
                        "subtype": "date_format",
                        "severity": "low",
                        "message": (
                            f"Column '{col}' contains date-like strings but is stored as object type. "
                            "Consider converting to datetime."
                        ),
                    })
        
        # Check string patterns (email, phone, etc.) - basic checks
        if series.dtype == "object":
            sample = series.dropna().head(100)
            if len(sample) > 0:
                # Check for email-like patterns
                email_like = sample.astype(str).str.contains(r"@", na=False).sum()
                if email_like > len(sample) * 0.8:
                    validation_results["checks"].append({
                        "column": col,
                        "type": "format",
                        "subtype": "email_pattern",
                        "severity": "info",
                        "message": f"Column '{col}' appears to contain email addresses",
                    })
    
    return validation_results


def validate_type_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate type consistency (mixed types, coercion issues).
    
    Returns:
        Dictionary with type consistency validation results
    """
    validation_results = {
        "checks": [],
        "summary": {
            "columns_with_type_issues": 0,
        },
    }
    
    for col in df.columns:
        series = df[col]
        
        # For object columns, check for mixed types
        if series.dtype == "object":
            sample = series.dropna().head(1000)
            if len(sample) > 0:
                # Try to detect mixed numeric/non-numeric
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                
                numeric_ratio = numeric_count / len(sample)
                
                # If mix of numeric and non-numeric, flag it
                if 0.1 < numeric_ratio < 0.9:
                    validation_results["summary"]["columns_with_type_issues"] += 1
                    
                    validation_results["checks"].append({
                        "column": col,
                        "type": "type_consistency",
                        "severity": "medium",
                        "numeric_ratio": float(numeric_ratio),
                        "message": (
                            f"Column '{col}' contains mixed numeric and non-numeric values "
                            f"({numeric_ratio:.1%} numeric). Consider data cleaning."
                        ),
                    })
    
    return validation_results


def run_full_validation(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all validation checks on a DataFrame.
    
    Args:
        df: DataFrame to validate
        key_columns: Optional list of columns that should be unique
    
    Returns:
        Complete validation report with all checks and summary
    """
    validation_report = {
        "table": "unknown",
        "checks": [],
        "summary": {
            "total_checks": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
        },
    }
    
    # Run all validation checks
    completeness = validate_completeness(df)
    uniqueness = validate_uniqueness(df, key_columns)
    ranges = validate_ranges(df)
    formats = validate_formats(df)
    types = validate_type_consistency(df)
    
    # Combine all checks
    all_checks = (
        completeness["checks"] +
        uniqueness["checks"] +
        ranges["checks"] +
        formats["checks"] +
        types["checks"]
    )
    
    validation_report["checks"] = all_checks
    
    # Count by severity
    for check in all_checks:
        severity = check.get("severity", "low")
        if severity == "critical":
            validation_report["summary"]["critical_issues"] += 1
        elif severity == "high":
            validation_report["summary"]["high_issues"] += 1
        elif severity == "medium":
            validation_report["summary"]["medium_issues"] += 1
        else:
            validation_report["summary"]["low_issues"] += 1
    
    validation_report["summary"]["total_checks"] = len(all_checks)
    
    return validation_report





